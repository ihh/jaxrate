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
