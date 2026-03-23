/**
 * WGSL shader source strings.
 * These are inlined so the package works without file I/O.
 * In production, a build step could generate this from .wgsl files.
 */

export const FORWARD_SHADER = /* wgsl */`
const NEG_INF: f32 = -1e38;

struct Params {
    K: u32,
    C: u32,
    col: u32,
    _pad: u32,
}

@group(0) @binding(0) var<storage, read> log_trans: array<f32>;
@group(0) @binding(1) var<storage, read> log_emit: array<f32>;
@group(0) @binding(2) var<storage, read> alpha_prev: array<f32>;
@group(0) @binding(3) var<storage, read_write> alpha_curr: array<f32>;
@group(0) @binding(4) var<uniform> params: Params;

var<workgroup> shared_vals: array<f32, 256>;

fn logaddexp(a: f32, b: f32) -> f32 {
    if (a <= NEG_INF + 1e30) { return b; }
    if (b <= NEG_INF + 1e30) { return a; }
    let m = max(a, b);
    return m + log(exp(a - m) + exp(b - m));
}

@compute @workgroup_size(256)
fn forward_step(
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(workgroup_id) wid: vec3<u32>,
) {
    let K = params.K;
    let col = params.col;
    let j = wid.x;
    if (j >= K) { return; }

    let local_id = lid.x;
    let wg_size = 256u;

    var partial = NEG_INF;
    var i = local_id;
    while (i < K) {
        partial = logaddexp(partial, log_trans[i * K + j] + alpha_prev[i]);
        i += wg_size;
    }

    shared_vals[local_id] = partial;
    workgroupBarrier();

    var stride = wg_size / 2u;
    while (stride > 0u) {
        if (local_id < stride) {
            shared_vals[local_id] = logaddexp(shared_vals[local_id], shared_vals[local_id + stride]);
        }
        workgroupBarrier();
        stride /= 2u;
    }

    if (local_id == 0u) {
        alpha_curr[j] = shared_vals[0] + log_emit[j * params.C + col];
    }
}
`;

export const BACKWARD_SHADER = /* wgsl */`
const NEG_INF: f32 = -1e38;

struct Params {
    K: u32,
    C: u32,
    col: u32,
    next_col: u32,
}

@group(0) @binding(0) var<storage, read> log_trans: array<f32>;
@group(0) @binding(1) var<storage, read> log_emit: array<f32>;
@group(0) @binding(2) var<storage, read> beta_next: array<f32>;
@group(0) @binding(3) var<storage, read_write> beta_curr: array<f32>;
@group(0) @binding(4) var<uniform> params: Params;

var<workgroup> shared_vals: array<f32, 256>;

fn logaddexp(a: f32, b: f32) -> f32 {
    if (a <= NEG_INF + 1e30) { return b; }
    if (b <= NEG_INF + 1e30) { return a; }
    let m = max(a, b);
    return m + log(exp(a - m) + exp(b - m));
}

@compute @workgroup_size(256)
fn backward_step(
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(workgroup_id) wid: vec3<u32>,
) {
    let K = params.K;
    let next_col = params.next_col;
    let i = wid.x;
    if (i >= K) { return; }

    let local_id = lid.x;
    let wg_size = 256u;

    var partial = NEG_INF;
    var j = local_id;
    while (j < K) {
        partial = logaddexp(partial, log_trans[i * K + j] + log_emit[j * params.C + next_col] + beta_next[j]);
        j += wg_size;
    }

    shared_vals[local_id] = partial;
    workgroupBarrier();

    var stride = wg_size / 2u;
    while (stride > 0u) {
        if (local_id < stride) {
            shared_vals[local_id] = logaddexp(shared_vals[local_id], shared_vals[local_id + stride]);
        }
        workgroupBarrier();
        stride /= 2u;
    }

    if (local_id == 0u) {
        beta_curr[i] = shared_vals[0];
    }
}
`;

export const VITERBI_SHADER = /* wgsl */`
const NEG_INF: f32 = -1e38;

struct Params {
    K: u32,
    C: u32,
    col: u32,
    _pad: u32,
}

@group(0) @binding(0) var<storage, read> log_trans: array<f32>;
@group(0) @binding(1) var<storage, read> log_emit: array<f32>;
@group(0) @binding(2) var<storage, read> v_prev: array<f32>;
@group(0) @binding(3) var<storage, read_write> v_curr: array<f32>;
@group(0) @binding(4) var<storage, read_write> bp_curr: array<u32>;
@group(0) @binding(5) var<uniform> params: Params;

var<workgroup> shared_vals: array<f32, 256>;
var<workgroup> shared_idx: array<u32, 256>;

@compute @workgroup_size(256)
fn viterbi_step(
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(workgroup_id) wid: vec3<u32>,
) {
    let K = params.K;
    let col = params.col;
    let j = wid.x;
    if (j >= K) { return; }

    let local_id = lid.x;
    let wg_size = 256u;

    var best_val = NEG_INF;
    var best_idx = 0u;
    var i = local_id;
    while (i < K) {
        let score = v_prev[i] + log_trans[i * K + j];
        if (score > best_val) {
            best_val = score;
            best_idx = i;
        }
        i += wg_size;
    }

    shared_vals[local_id] = best_val;
    shared_idx[local_id] = best_idx;
    workgroupBarrier();

    var stride = wg_size / 2u;
    while (stride > 0u) {
        if (local_id < stride) {
            if (shared_vals[local_id + stride] > shared_vals[local_id]) {
                shared_vals[local_id] = shared_vals[local_id + stride];
                shared_idx[local_id] = shared_idx[local_id + stride];
            }
        }
        workgroupBarrier();
        stride /= 2u;
    }

    if (local_id == 0u) {
        v_curr[j] = shared_vals[0] + log_emit[j * params.C + col];
        bp_curr[j] = shared_idx[0];
    }
}
`;

export const SCFG_INSIDE_SHADER = /* wgsl */`
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

@group(0) @binding(0) var<storage, read_write> alpha: array<f32>;
@group(0) @binding(1) var<storage, read> single_emit: array<f32>;
@group(0) @binding(2) var<storage, read> paired_emit: array<f32>;
@group(0) @binding(3) var<storage, read> rules: array<u32>;
@group(0) @binding(4) var<uniform> params: Params;

fn rule_field(rule_idx: u32, field: u32) -> u32 {
    return rules[rule_idx * 6u + field];
}

fn rule_log_weight(rule_idx: u32) -> f32 {
    return bitcast<f32>(rules[rule_idx * 6u + 5u]);
}

@compute @workgroup_size(1)
fn emit_step(@builtin(workgroup_id) wid: vec3<u32>) {
    let S = params.S;
    let C = params.C;
    let span = params.span;
    let i = wid.x;
    let j = i + span;
    if (j > C) { return; }

    if (span == 1u) {
        let base = 0u;
        for (var r = 0u; r < params.n_terminal; r++) {
            let lhs = rule_field(base + r, 0u);
            let model = rule_field(base + r, 3u);
            let log_w = rule_log_weight(base + r);
            let idx = chart_idx(lhs, i, j, S);
            alpha[idx] = logaddexp(alpha[idx], log_w + single_emit[model * C + i]);
        }
    }

    let eu_base = params.n_terminal;
    for (var r = 0u; r < params.n_emit_unary; r++) {
        let lhs = rule_field(eu_base + r, 0u);
        let rhs = rule_field(eu_base + r, 1u);
        let model = rule_field(eu_base + r, 3u);
        let log_w = rule_log_weight(eu_base + r);
        let child_val = alpha[chart_idx(rhs, i + 1u, j, S)];
        if (child_val > NEG_INF + 1e30) {
            let idx = chart_idx(lhs, i, j, S);
            alpha[idx] = logaddexp(alpha[idx], log_w + single_emit[model * C + i] + child_val);
        }
    }

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

    var stride = wg_size / 2u;
    while (stride > 0u) {
        if (local_id < stride) {
            shared_vals[local_id] = logaddexp(shared_vals[local_id], shared_vals[local_id + stride]);
        }
        workgroupBarrier();
        stride /= 2u;
    }

    if (local_id == 0u) {
        let idx = chart_idx(lhs, i, j, S);
        alpha[idx] = logaddexp(alpha[idx], shared_vals[0]);
    }
}

@compute @workgroup_size(1)
fn unary_propagate(@builtin(workgroup_id) wid: vec3<u32>) {
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
`;

export const SCFG_VITERBI_SHADER = /* wgsl */`
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

@compute @workgroup_size(1)
fn emit_step(@builtin(workgroup_id) wid: vec3<u32>) {
    let S = params.S;
    let C = params.C;
    let span = params.span;
    let i = wid.x;
    let j = i + span;
    if (j > C) { return; }

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

@compute @workgroup_size(1)
fn unary_propagate(@builtin(workgroup_id) wid: vec3<u32>) {
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
`;
