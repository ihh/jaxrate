// HMM Forward algorithm — diagonal sweep.
//
// Processing order: columns are sequential (c = 0, 1, ..., C-1).
// Within each column, all K destination states are computed in parallel.
//
// Each dispatch processes ONE column. The host dispatches C times sequentially,
// using ping-pong buffers (alpha_prev ↔ alpha_curr).
//
// Workgroup layout: each invocation handles one destination state j.
// The logsumexp over source states i is computed via workgroup reduction.
//
// Bindings:
//   @group(0) @binding(0) log_trans:  (K, K) row-major — log_trans[i * K + j]
//   @group(0) @binding(1) log_emit:   (K, C) row-major — log_emit[k * C + c]
//   @group(0) @binding(2) alpha_prev: (K,) — alpha values from previous column
//   @group(0) @binding(3) alpha_curr: (K,) — output alpha values for current column
//   @group(0) @binding(4) params:     {K, C, col} — dimensions and current column

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

// Shared memory for workgroup reduction
var<workgroup> shared_vals: array<f32, 256>;

fn logaddexp(a: f32, b: f32) -> f32 {
    if (a <= NEG_INF + 1e30) { return b; }
    if (b <= NEG_INF + 1e30) { return a; }
    let m = max(a, b);
    return m + log(exp(a - m) + exp(b - m));
}

@compute @workgroup_size(256)
fn forward_step(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(workgroup_id) wid: vec3<u32>,
) {
    let K = params.K;
    let col = params.col;

    // Each workgroup handles one destination state j
    let j = wid.x;
    if (j >= K) { return; }

    let local_id = lid.x;
    let workgroup_size = 256u;

    // Phase 1: Each thread computes partial logsumexp over a chunk of source states
    var partial = NEG_INF;
    var i = local_id;
    while (i < K) {
        let trans_val = log_trans[i * K + j];
        let prev_val = alpha_prev[i];
        partial = logaddexp(partial, trans_val + prev_val);
        i += workgroup_size;
    }

    shared_vals[local_id] = partial;
    workgroupBarrier();

    // Phase 2: Parallel reduction in shared memory
    var stride = workgroup_size / 2u;
    while (stride > 0u) {
        if (local_id < stride) {
            shared_vals[local_id] = logaddexp(shared_vals[local_id], shared_vals[local_id + stride]);
        }
        workgroupBarrier();
        stride = stride / 2u;
    }

    // Phase 3: Thread 0 writes the result
    if (local_id == 0u) {
        let emit_val = log_emit[j * params.C + col];
        alpha_curr[j] = shared_vals[0] + emit_val;
    }
}
