"""Pre-computation bridge: subby → terminal weight tables."""

import jax.numpy as jnp
import numpy as np

from .types import TerminalWeights


def precompute_terminal_weights(alignment, tree, models,
                                paired_models=None, kmer_models=None,
                                maxChunkSize=128):
    """Pre-compute all terminal weights for grammar algorithms.

    Computes phylogenetic log-likelihoods under each model for each column
    (or column pair/k-mer) once upfront. Grammar DP then does table lookups.

    Args:
        alignment: (R, C) int32 token-encoded alignment
        tree: subby Tree or dict with parentIndex and distanceToParent
        models: list of single-column subby models (DiagModel or dict)
        paired_models: optional list of paired-column models. Each entry is
                       a dict with 'model' and 'A' keys. The paired model
                       operates on A*A joint states.
        kmer_models: optional list of dicts with 'model', 'k', 'stride', 'A' keys.
                     k-mer terminal weights via sliding_windows + kmer_tokenize.
        maxChunkSize: column chunk size for subby LogLike

    Returns:
        TerminalWeights NamedTuple
    """
    from subby.jax import LogLike

    C = alignment.shape[1]

    # Single-column weights: (K_single, C)
    single_lls = []
    for model in models:
        ll = LogLike(alignment, tree, model, maxChunkSize=maxChunkSize)
        single_lls.append(ll)
    single = jnp.stack(single_lls, axis=0)  # (K_single, C)

    # Paired-column weights: (K_paired, C, C)
    paired = None
    if paired_models is not None and len(paired_models) > 0:
        from subby.formats import kmer_tokenize, all_column_ktuples

        paired_lls = []
        for pm in paired_models:
            p_model = pm['model']
            p_A = pm.get('A', 20)
            # Generate all ordered column pairs
            tuples = all_column_ktuples(C, 2, ordered=True)
            if len(tuples) == 0:
                paired_lls.append(jnp.zeros((C, C)))
                continue
            kt = kmer_tokenize(np.asarray(alignment), p_A, tuples, gap_mode='any')
            # kt['alignment'] is (R, T) with A_kmer = p_A^2
            # Run LogLike on the paired alignment
            ll = LogLike(jnp.array(kt['alignment']), tree, p_model,
                         maxChunkSize=maxChunkSize)
            # Reshape (T,) → (C, C)
            ll_matrix = jnp.full((C, C), -1e38)
            for t_idx in range(len(tuples)):
                ci, cj = tuples[t_idx]
                ll_matrix = ll_matrix.at[ci, cj].set(ll[t_idx])
            paired_lls.append(ll_matrix)
        paired = jnp.stack(paired_lls, axis=0)  # (K_paired, C, C)

    # K-mer weights: (K_kmer, C_kmer)
    kmer = None
    if kmer_models is not None and len(kmer_models) > 0:
        from subby.formats import kmer_tokenize, sliding_windows

        kmer_lls = []
        for km in kmer_models:
            k_model = km['model']
            k_size = km['k']
            k_stride = km.get('stride', 1)
            k_A = km.get('A', 4)
            windows = sliding_windows(C, k_size, stride=k_stride)
            if len(windows) == 0:
                kmer_lls.append(jnp.zeros((0,)))
                continue
            kt = kmer_tokenize(np.asarray(alignment), k_A, windows, gap_mode='any')
            ll = LogLike(jnp.array(kt['alignment']), tree, k_model,
                         maxChunkSize=maxChunkSize)
            kmer_lls.append(ll)
        kmer = jnp.stack(kmer_lls, axis=0) if kmer_lls else None

    return TerminalWeights(
        single=single,
        paired=paired,
        kmer=kmer,
        C=C,
    )
