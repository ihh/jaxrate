// SCFG Viterbi/CYK — span-level wavefront with max instead of logsumexp.
//
// Same dispatch pattern as inside: one dispatch per span length.
// Tracks backpointers for traceback on CPU.
//
// Backpointer layout: bp[A * S * S + i * S + j] = packed u32:
//   bits [0:15]  = rule index
//   bits [16:31] = split point k (for binary rules, else 0)

const NEG_INF: f32 = -1e38;

fn chart_idx(A: u32, i: u32, j: u32, S: u32) -> u32 {
    return A * S * S + i * S + j;
}

fn pack_bp(rule: u32, split: u32) -> u32 {
    return (split << 16u) | (rule & 0xFFFFu);
}

struct Params {
    K: u32,
    C: u32,
    S: u32,
    span: u32,
    n_terminal: u32,
    n_emit_unary: u32,
    n_emit_paired: u32,
    n_binary: u32,
    n_unary: u32,
    K_paired: u32,
    _pad0: u32,
    _pad1: u32,
}

@group(0) @binding(0) var<storage, read_write> v: array<f32>;
@group(0) @binding(1) var<storage, read> single_emit: array<f32>;
@group(0) @binding(2) var<storage, read> paired_emit: array<f32>;
@group(0) @binding(3) var<storage, read> rules: array<u32>;
@group(0) @binding(4) var<storage, read_write> bp: array<u32>;
@group(0) @binding(5) var<uniform> params: Params;

fn rule_field(rule_idx: u32, field: u32) -> u32 {
    return rules[rule_idx * 6u + field];
}

fn rule_log_weight(rule_idx: u32) -> f32 {
    return bitcast<f32>(rules[rule_idx * 6u + 5u]);
}

// ── Kernel 1: Terminal + emission rules (max) ───────────────────

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

    // Terminal (span 1)
    if (span == 1u) {
        for (var r = 0u; r < params.n_terminal; r++) {
            let lhs = rule_field(r, 0u);
            let model = rule_field(r, 3u);
            let log_w = rule_log_weight(r);
            let score = log_w + single_emit[model * C + i];
            let idx = chart_idx(lhs, i, j, S);
            if (score > v[idx]) {
                v[idx] = score;
                bp[idx] = pack_bp(r, 0u);
            }
        }
    }

    // Left-emit unary
    let eu_base = params.n_terminal;
    for (var r = 0u; r < params.n_emit_unary; r++) {
        let ri = eu_base + r;
        let lhs = rule_field(ri, 0u);
        let rhs = rule_field(ri, 1u);
        let model = rule_field(ri, 3u);
        let log_w = rule_log_weight(ri);
        let child_val = v[chart_idx(rhs, i + 1u, j, S)];
        if (child_val > NEG_INF + 1e30) {
            let score = log_w + single_emit[model * C + i] + child_val;
            let idx = chart_idx(lhs, i, j, S);
            if (score > v[idx]) {
                v[idx] = score;
                bp[idx] = pack_bp(ri, 0u);
            }
        }
    }

    // Paired emission (span >= 2)
    if (span >= 2u && params.K_paired > 0u) {
        let ep_base = params.n_terminal + params.n_emit_unary;
        for (var r = 0u; r < params.n_emit_paired; r++) {
            let ri = ep_base + r;
            let lhs = rule_field(ri, 0u);
            let rhs = rule_field(ri, 1u);
            let model = rule_field(ri, 3u);
            let log_w = rule_log_weight(ri);
            let pair_w = paired_emit[model * C * C + i * C + (j - 1u)];
            let child_val = v[chart_idx(rhs, i + 1u, j - 1u, S)];
            if (child_val > NEG_INF + 1e30) {
                let score = log_w + pair_w + child_val;
                let idx = chart_idx(lhs, i, j, S);
                if (score > v[idx]) {
                    v[idx] = score;
                    bp[idx] = pack_bp(ri, 0u);
                }
            }
        }
    }
}

// ── Kernel 2: Binary rules — split-point max reduction ──────────

var<workgroup> shared_vals: array<f32, 256>;
var<workgroup> shared_split: array<u32, 256>;

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
    let ri = bin_base + rule_local;
    let lhs = rule_field(ri, 0u);
    let rhs_b = rule_field(ri, 1u);
    let rhs_c = rule_field(ri, 2u);
    let log_w = rule_log_weight(ri);

    let local_id = lid.x;
    let wg_size = 256u;

    var best_val = NEG_INF;
    var best_k = i;
    var k = i + local_id;
    while (k <= j) {
        let left_val = v[chart_idx(rhs_b, i, k, S)];
        let right_val = v[chart_idx(rhs_c, k, j, S)];
        if (left_val > NEG_INF + 1e30 && right_val > NEG_INF + 1e30) {
            let score = log_w + left_val + right_val;
            if (score > best_val) {
                best_val = score;
                best_k = k;
            }
        }
        k += wg_size;
    }

    shared_vals[local_id] = best_val;
    shared_split[local_id] = best_k;
    workgroupBarrier();

    var stride = wg_size / 2u;
    while (stride > 0u) {
        if (local_id < stride) {
            if (shared_vals[local_id + stride] > shared_vals[local_id]) {
                shared_vals[local_id] = shared_vals[local_id + stride];
                shared_split[local_id] = shared_split[local_id + stride];
            }
        }
        workgroupBarrier();
        stride /= 2u;
    }

    if (local_id == 0u) {
        let idx = chart_idx(lhs, i, j, S);
        if (shared_vals[0] > v[idx]) {
            v[idx] = shared_vals[0];
            bp[idx] = pack_bp(ri, shared_split[0]);
        }
    }
}

// ── Kernel 3: Unary propagation (max) ───────────────────────────

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

    for (var pass = 0u; pass < params.K; pass++) {
        var changed = false;
        for (var r = 0u; r < params.n_unary; r++) {
            let ri = un_base + r;
            let lhs = rule_field(ri, 0u);
            let rhs = rule_field(ri, 1u);
            let log_w = rule_log_weight(ri);
            let child_val = v[chart_idx(rhs, i, j, S)];
            if (child_val > NEG_INF + 1e30) {
                let score = log_w + child_val;
                let idx = chart_idx(lhs, i, j, S);
                if (score > v[idx] + 1e-6) {
                    v[idx] = score;
                    bp[idx] = pack_bp(ri, 0u);
                    changed = true;
                }
            }
        }
        if (!changed) { break; }
    }
}
