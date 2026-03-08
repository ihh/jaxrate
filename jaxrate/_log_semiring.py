"""Log-semiring utilities for numerically stable grammar computations."""

import jax
import jax.numpy as jnp

NEG_INF = -1e38  # Practical -inf that avoids NaN in gradients


def logsumexp(a, axis=None, keepdims=False):
    """Numerically stable logsumexp."""
    return jax.scipy.special.logsumexp(a, axis=axis, keepdims=keepdims)


def logaddexp(a, b):
    """Numerically stable log(exp(a) + exp(b))."""
    return jnp.logaddexp(a, b)


def log_matmul(log_A, log_B):
    """Matrix multiply in log space: log(exp(A) @ exp(B)).

    Args:
        log_A: (..., M, K) log-probabilities
        log_B: (..., K, N) log-probabilities

    Returns:
        (..., M, N) log-probabilities
    """
    # log_A[..., :, :, None] + log_B[..., None, :, :] → (..., M, K, N)
    return logsumexp(log_A[..., :, :, None] + log_B[..., None, :, :], axis=-2)


def log_matvec(log_A, log_x):
    """Matrix-vector multiply in log space: log(exp(A) @ exp(x)).

    Args:
        log_A: (..., M, K) log-probabilities
        log_x: (..., K) log-probabilities

    Returns:
        (..., M) log-probabilities
    """
    return logsumexp(log_A + log_x[..., None, :], axis=-1)
