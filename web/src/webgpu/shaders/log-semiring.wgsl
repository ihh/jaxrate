// Log-semiring utilities for WebGPU compute shaders.
// f32 arithmetic throughout (WebGPU f64 not universally supported).

const NEG_INF: f32 = -1e38;

// Numerically stable log(exp(a) + exp(b))
fn logaddexp(a: f32, b: f32) -> f32 {
    if (a <= NEG_INF + 1e30) { return b; }
    if (b <= NEG_INF + 1e30) { return a; }
    let m = max(a, b);
    return m + log(exp(a - m) + exp(b - m));
}
