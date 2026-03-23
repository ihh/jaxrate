// SCFG Inside algorithm — span-level wavefront.
//
// Host dispatches once per span length (1..C), sequentially.
// Within each span, all (i, rule) combinations execute in parallel.
//
// Three kernels per span dispatch:
//   1. emit_and_unary_step: terminal, left-emit, paired rules (no split points)
//   2. binary_step: binary rules with split-point reduction
//   3. unary_propagate: null unary chain propagation (multiple passes)
//
// Chart layout: alpha[A * S * S + i * S + j], S = C+1, row-major.
//
// Bindings:
//   @binding(0) alpha:        (K * S * S) chart — read_write
//   @binding(1) single_emit:  (K_single * C) single terminal weights
//   @binding(2) paired_emit:  (K_paired * C * C) paired terminal weights
//   @binding(3) rules:        packed rule descriptors
//   @binding(4) params:       {K, C, S, span, n_*_rules, ...}

const NEG_INF: f32 = -1e38;

fn logaddexp(a: f32, b: f32) -> f32 {
    if (a <= NEG_INF + 1e30) { return b; }
    if (b <= NEG_INF + 1e30) { return a; }
    let m = max(a, b);
    return m + log(exp(a - m) + exp(b - m));
}

fn chart_idx(A: u32, i: u32, j: u32, S: u32) -> u32 {
    return A * S * S + i * S + j;
}

// ── Rule descriptor layout ────────────────────────────────────────
// Each rule is packed into 6 u32s:
//   [0] lhs
//   [1] rhs0  (-1 if none)
//   [2] rhs1  (-1 if none)
//   [3] emit_model (-1 if none)
//   [4] emit_npos  (0, 1, or 2)
//   [5] log_weight (as bitcast f32)

struct Params {
    K: u32,
    C: u32,
    S: u32,          // C + 1
    span: u32,
    n_terminal: u32,
    n_emit_unary: u32,
    n_emit_paired: u32,
    n_binary: u32,
    n_unary: u32,
    K_paired: u32,   // number of paired models (0 if none)
    _pad0: u32,
    _pad1: u32,
}

@group(0) @binding(0) var<storage, read_write> alpha: array<f32>;
@group(0) @binding(1) var<storage, read> single_emit: array<f32>;
@group(0) @binding(2) var<storage, read> paired_emit: array<f32>;
@group(0) @binding(3) var<storage, read> rules: array<u32>;
@group(0) @binding(4) var<uniform> params: Params;

// Helper to read a rule field
fn rule_field(rule_idx: u32, field: u32) -> u32 {
    return rules[rule_idx * 6u + field];
}

fn rule_log_weight(rule_idx: u32) -> f32 {
    return bitcast<f32>(rules[rule_idx * 6u + 5u]);
}

// ── Kernel 1: Terminal + emission rules ─────────────────────────
// Dispatch: (C - span + 1) workgroups, 1 thread each.
// Each workgroup handles position i for all terminal/emit rules.

@compute @workgroup_size(1)
fn emit_step(
    @builtin(workgroup_id) wid: vec3<u32>,
) {
    let S = params.S;
    let C = params.C;
    let span = params.span;
    let i = wid.x;
    let j = i + span;

    if (j > C) { return; }

    // Terminal rules (span 1 only)
    if (span == 1u) {
        let base = 0u;  // terminal rules start at offset 0
        for (var r = 0u; r < params.n_terminal; r++) {
            let lhs = rule_field(base + r, 0u);
            let model = rule_field(base + r, 3u);
            let log_w = rule_log_weight(base + r);
            let emit_w = single_emit[model * C + i];
            let idx = chart_idx(lhs, i, j, S);
            alpha[idx] = logaddexp(alpha[idx], log_w + emit_w);
        }
    }

    // Left-emit unary rules: A → e B
    let eu_base = params.n_terminal;
    for (var r = 0u; r < params.n_emit_unary; r++) {
        let lhs = rule_field(eu_base + r, 0u);
        let rhs = rule_field(eu_base + r, 1u);
        let model = rule_field(eu_base + r, 3u);
        let log_w = rule_log_weight(eu_base + r);
        let emit_w = single_emit[model * C + i];
        let child_val = alpha[chart_idx(rhs, i + 1u, j, S)];
        if (child_val > NEG_INF + 1e30) {
            let idx = chart_idx(lhs, i, j, S);
            alpha[idx] = logaddexp(alpha[idx], log_w + emit_w + child_val);
        }
    }

    // Paired emission rules: A → e₁ B e₂ (span >= 2)
    if (span >= 2u && params.K_paired > 0u) {
        let ep_base = params.n_terminal + params.n_emit_unary;
        for (var r = 0u; r < params.n_emit_paired; r++) {
            let lhs = rule_field(ep_base + r, 0u);
            let rhs = rule_field(ep_base + r, 1u);
            let model = rule_field(ep_base + r, 3u);
            let log_w = rule_log_weight(ep_base + r);
            let pair_w = paired_emit[model * C * C + i * C + (j - 1u)];
            let child_val = alpha[chart_idx(rhs, i + 1u, j - 1u, S)];
            if (child_val > NEG_INF + 1e30) {
                let idx = chart_idx(lhs, i, j, S);
                alpha[idx] = logaddexp(alpha[idx], log_w + pair_w + child_val);
            }
        }
    }
}

// ── Kernel 2: Binary rules with split-point reduction ───────────
// Dispatch: (C - span + 1) × n_binary workgroups, 256 threads each.
// wid.x = position i, wid.y = binary rule index.
// Each workgroup reduces over split points k ∈ [i, j].

var<workgroup> shared_vals: array<f32, 256>;

@compute @workgroup_size(256)
fn binary_step(
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(workgroup_id) wid: vec3<u32>,
) {
    let S = params.S;
    let C = params.C;
    let span = params.span;
    let i = wid.x;
    let j = i + span;
    let rule_local = wid.y;

    if (j > C || rule_local >= params.n_binary) { return; }

    let bin_base = params.n_terminal + params.n_emit_unary + params.n_emit_paired;
    let r = bin_base + rule_local;
    let lhs = rule_field(r, 0u);
    let rhs_b = rule_field(r, 1u);
    let rhs_c = rule_field(r, 2u);
    let log_w = rule_log_weight(r);

    let local_id = lid.x;
    let wg_size = 256u;

    // Each thread handles a subset of split points
    var partial = NEG_INF;
    var k = i + local_id;
    while (k <= j) {
        let left_val = alpha[chart_idx(rhs_b, i, k, S)];
        let right_val = alpha[chart_idx(rhs_c, k, j, S)];
        if (left_val > NEG_INF + 1e30 && right_val > NEG_INF + 1e30) {
            partial = logaddexp(partial, log_w + left_val + right_val);
        }
        k += wg_size;
    }

    shared_vals[local_id] = partial;
    workgroupBarrier();

    // Parallel reduction
    var stride = wg_size / 2u;
    while (stride > 0u) {
        if (local_id < stride) {
            shared_vals[local_id] = logaddexp(shared_vals[local_id], shared_vals[local_id + stride]);
        }
        workgroupBarrier();
        stride /= 2u;
    }

    // Thread 0 accumulates into the chart (atomic-free since one workgroup per (i, rule))
    if (local_id == 0u) {
        let idx = chart_idx(lhs, i, j, S);
        alpha[idx] = logaddexp(alpha[idx], shared_vals[0]);
    }
}

// ── Kernel 3: Unary propagation ─────────────────────────────────
// Dispatch: (C - span + 1) workgroups, 1 thread each.
// Runs multiple passes to propagate through unary chains.

@compute @workgroup_size(1)
fn unary_propagate(
    @builtin(workgroup_id) wid: vec3<u32>,
) {
    let S = params.S;
    let C = params.C;
    let span = params.span;
    let i = wid.x;
    let j = i + span;

    if (j > C) { return; }

    let un_base = params.n_terminal + params.n_emit_unary + params.n_emit_paired + params.n_binary;

    // Multiple passes for chain propagation
    for (var pass = 0u; pass < params.K; pass++) {
        var changed = false;
        for (var r = 0u; r < params.n_unary; r++) {
            let lhs = rule_field(un_base + r, 0u);
            let rhs = rule_field(un_base + r, 1u);
            let log_w = rule_log_weight(un_base + r);
            let child_val = alpha[chart_idx(rhs, i, j, S)];
            if (child_val > NEG_INF + 1e30) {
                let idx = chart_idx(lhs, i, j, S);
                let old_val = alpha[idx];
                let new_val = logaddexp(old_val, log_w + child_val);
                if (new_val > old_val + 1e-6) {
                    alpha[idx] = new_val;
                    changed = true;
                }
            }
        }
        if (!changed) { break; }
    }
}
