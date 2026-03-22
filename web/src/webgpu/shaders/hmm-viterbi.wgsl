// HMM Viterbi algorithm — diagonal sweep with max instead of logsumexp.
//
// Each dispatch processes ONE column.
// v[c, j] = max_i(v[c-1, i] + log_trans[i, j]) + log_emit[j, c]
// bp[c, j] = argmax_i(v[c-1, i] + log_trans[i, j])
//
// Each workgroup handles one destination state j.
// Parallel reduction for max + argmax over source states i.

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

// Shared memory: values and indices for max reduction
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
    let workgroup_size = 256u;

    // Phase 1: Each thread finds max over its chunk of source states
    var best_val = NEG_INF;
    var best_idx = 0u;
    var i = local_id;
    while (i < K) {
        let score = v_prev[i] + log_trans[i * K + j];
        if (score > best_val) {
            best_val = score;
            best_idx = i;
        }
        i += workgroup_size;
    }

    shared_vals[local_id] = best_val;
    shared_idx[local_id] = best_idx;
    workgroupBarrier();

    // Phase 2: Parallel reduction for max + argmax
    var stride = workgroup_size / 2u;
    while (stride > 0u) {
        if (local_id < stride) {
            if (shared_vals[local_id + stride] > shared_vals[local_id]) {
                shared_vals[local_id] = shared_vals[local_id + stride];
                shared_idx[local_id] = shared_idx[local_id + stride];
            }
        }
        workgroupBarrier();
        stride = stride / 2u;
    }

    // Phase 3: Thread 0 writes result
    if (local_id == 0u) {
        let emit_val = log_emit[j * params.C + col];
        v_curr[j] = shared_vals[0] + emit_val;
        bp_curr[j] = shared_idx[0];
    }
}
