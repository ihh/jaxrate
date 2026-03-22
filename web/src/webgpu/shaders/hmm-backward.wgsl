// HMM Backward algorithm — diagonal sweep (reverse direction).
//
// Each dispatch processes ONE column (from C-2 down to 0).
// beta[c, i] = logsumexp_j(log_trans[i, j] + log_emit[j, c+1] + beta[c+1, j])
//
// Each workgroup handles one source state i.
// Reduction over destination states j.

const NEG_INF: f32 = -1e38;

struct Params {
    K: u32,
    C: u32,
    col: u32,         // current column c
    next_col: u32,    // c + 1
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

    // Each workgroup handles one source state i
    let i = wid.x;
    if (i >= K) { return; }

    let local_id = lid.x;
    let workgroup_size = 256u;

    // Each thread handles a chunk of destination states j
    var partial = NEG_INF;
    var j = local_id;
    while (j < K) {
        let trans_val = log_trans[i * K + j];
        let emit_val = log_emit[j * params.C + next_col];
        let beta_val = beta_next[j];
        partial = logaddexp(partial, trans_val + emit_val + beta_val);
        j += workgroup_size;
    }

    shared_vals[local_id] = partial;
    workgroupBarrier();

    // Parallel reduction
    var stride = workgroup_size / 2u;
    while (stride > 0u) {
        if (local_id < stride) {
            shared_vals[local_id] = logaddexp(shared_vals[local_id], shared_vals[local_id + stride]);
        }
        workgroupBarrier();
        stride = stride / 2u;
    }

    if (local_id == 0u) {
        beta_curr[i] = shared_vals[0];
    }
}
