/**
 * Log-semiring arithmetic utilities.
 * All computations in log-space to prevent underflow.
 */

/** Practical negative infinity that avoids NaN in gradients. */
export const NEG_INF = -1e38;

/** Numerically stable log(exp(a) + exp(b)). */
export function logaddexp(a: number, b: number): number {
  if (a === NEG_INF) return b;
  if (b === NEG_INF) return a;
  const m = Math.max(a, b);
  return m + Math.log(Math.exp(a - m) + Math.exp(b - m));
}

/** Numerically stable log-sum-exp over an array. */
export function logsumexp(arr: Float64Array | number[], start = 0, end?: number): number {
  const n = end ?? arr.length;
  if (n - start <= 0) return NEG_INF;

  let max = NEG_INF;
  for (let i = start; i < n; i++) {
    if (arr[i] > max) max = arr[i];
  }
  if (max === NEG_INF) return NEG_INF;

  let sum = 0;
  for (let i = start; i < n; i++) {
    sum += Math.exp(arr[i] - max);
  }
  return max + Math.log(sum);
}

/**
 * Log-space matrix-vector multiply: result[j] = logsumexp_i(A[i,j] + x[i]).
 * A is (M, N) row-major, x is (M,), result is (N,).
 */
export function logMatvec(
  A: Float64Array, M: number, N: number,
  x: Float64Array,
  result: Float64Array
): void {
  const tmp = new Float64Array(M);
  for (let j = 0; j < N; j++) {
    for (let i = 0; i < M; i++) {
      tmp[i] = A[i * N + j] + x[i];
    }
    result[j] = logsumexp(tmp);
  }
}

/**
 * Log-space matrix-vector multiply (transposed): result[i] = logsumexp_j(A[i,j] + x[j]).
 * A is (M, N) row-major, x is (N,), result is (M,).
 */
export function logMatvecT(
  A: Float64Array, M: number, N: number,
  x: Float64Array,
  result: Float64Array
): void {
  const tmp = new Float64Array(N);
  for (let i = 0; i < M; i++) {
    for (let j = 0; j < N; j++) {
      tmp[j] = A[i * N + j] + x[j];
    }
    result[i] = logsumexp(tmp);
  }
}

/** Fill a Float64Array with NEG_INF. */
export function fillNegInf(arr: Float64Array): void {
  arr.fill(NEG_INF);
}

/** Create a new Float64Array filled with NEG_INF. */
export function negInfArray(length: number): Float64Array {
  const arr = new Float64Array(length);
  arr.fill(NEG_INF);
  return arr;
}
