"""HMM (right-linear grammar) algorithms using jax.lax.scan.

All algorithms operate in log-space for numerical stability.
Complexity: O(C * K²) where K = number of states, C = number of columns.
"""

import jax
import jax.numpy as jnp
from functools import partial

from ._log_semiring import NEG_INF, logsumexp


def _build_hmm_tables(cg, tw):
    """Extract HMM transition matrix and emission weights from compiled grammar.

    For an HMM (right-linear grammar), each rule is either:
    - Terminal: LHS → emit (no RHS nonterminal)
    - Transition: LHS → emit RHS

    We build:
    - log_trans: (K, K) log transition matrix (trans[i,j] = log P(i → j))
    - log_emit: (K, C) log emission weights per state per column
    - log_init: (K,) initial state log-probabilities
    - log_term: (K,) log-probability of terminating at each state

    For multi-column emissions (e.g., codons emitting 3 columns), the emission
    at column c uses the k-mer terminal weight covering that position.
    """
    K = cg.n_nonterminals
    C = tw.C

    log_trans = jnp.full((K, K), NEG_INF)
    log_term = jnp.full((K,), NEG_INF)
    log_emit_model = jnp.full((K,), -1, dtype=jnp.int32)  # which model each state uses
    emit_n_pos = jnp.ones((K,), dtype=jnp.int32)  # columns consumed per emission

    # Process rules to build transition matrix
    for r in range(cg.n_rules):
        lhs = int(cg.rule_lhs[r])
        n_rhs = int(cg.rule_n_rhs[r])
        log_w = float(cg.rule_log_weights[r])
        n_emit = int(cg.rule_n_emissions[r])

        if n_emit > 0:
            emit_model = int(cg.rule_emission_model[r, 0])
            emit_npos = int(cg.rule_emission_npos[r, 0])
            log_emit_model = log_emit_model.at[lhs].set(emit_model)
            emit_n_pos = emit_n_pos.at[lhs].set(emit_npos)

        if n_rhs == 0:
            # Terminal rule: LHS → emit
            log_term = log_term.at[lhs].set(
                jnp.logaddexp(log_term[lhs], log_w))
        else:
            # Transition rule: LHS → emit RHS[0]
            rhs = int(cg.rule_rhs[r, 0])
            log_trans = log_trans.at[lhs, rhs].set(
                jnp.logaddexp(log_trans[lhs, rhs], log_w))

    # Build emission table: (K, C)
    # For single-column states, emit[k, c] = tw.single[model_k, c]
    # For k-mer states, emit[k, c] uses tw.kmer if available
    log_emit = jnp.zeros((K, C))
    for k in range(K):
        model_idx = int(log_emit_model[k])
        if model_idx >= 0:
            npos = int(emit_n_pos[k])
            if npos == 1:
                log_emit = log_emit.at[k].set(tw.single[model_idx])
            elif tw.kmer is not None:
                # Multi-column emission: use k-mer terminal weights
                log_emit = log_emit.at[k].set(
                    jnp.pad(tw.kmer[model_idx],
                            (0, max(0, C - tw.kmer[model_idx].shape[0])),
                            constant_values=NEG_INF)[:C])

    # Initial state distribution: start nonterminal entered with probability 1.
    # The HMM begins in the start state, emits, then transitions.
    log_init = jnp.full((K,), NEG_INF)
    log_init = log_init.at[cg.start].set(0.0)

    return log_trans, log_emit, log_init, log_term


def hmm_forward(cg, tw):
    """Forward algorithm (scan-based).

    Computes log P(x_{1:c}, state_c = k) for all c, k.

    Args:
        cg: CompiledGrammar (must be 'hmm' class)
        tw: TerminalWeights

    Returns:
        alpha: (C, K) log forward probabilities
        log_likelihood: scalar log P(x_{1:C})
    """
    log_trans, log_emit, log_init, log_term = _build_hmm_tables(cg, tw)
    K = cg.n_nonterminals
    C = tw.C

    # alpha[0, k] = log_init[k] + log_emit[k, 0]
    alpha_0 = log_init + log_emit[:, 0]

    def scan_fn(alpha_prev, c):
        # alpha[c, j] = logsumexp_i(alpha[c-1, i] + log_trans[i, j]) + log_emit[j, c]
        # log_trans[i, j] is (K, K), alpha_prev is (K,)
        # alpha_prev[:, None] + log_trans → (K, K), logsumexp over axis 0 → (K,)
        alpha_c = logsumexp(alpha_prev[:, None] + log_trans, axis=0) + log_emit[:, c]
        return alpha_c, alpha_c

    _, alphas = jax.lax.scan(scan_fn, alpha_0, jnp.arange(1, C))

    # Stack: alpha_0 + rest
    alpha = jnp.concatenate([alpha_0[None, :], alphas], axis=0)  # (C, K)

    # Log-likelihood: logsumexp over terminal states at last column
    log_likelihood = logsumexp(alpha[-1] + log_term)

    return alpha, log_likelihood


def hmm_backward(cg, tw):
    """Backward algorithm (reverse scan).

    Computes log P(x_{c+1:C} | state_c = k) for all c, k.

    Args:
        cg: CompiledGrammar
        tw: TerminalWeights

    Returns:
        beta: (C, K) log backward probabilities
        log_likelihood: scalar log P(x_{1:C})
    """
    log_trans, log_emit, log_init, log_term = _build_hmm_tables(cg, tw)
    K = cg.n_nonterminals
    C = tw.C

    # beta[C-1, k] = log_term[k]
    beta_last = log_term

    def scan_fn(beta_next, c):
        # beta[c, i] = logsumexp_j(log_trans[i, j] + log_emit[j, c+1] + beta[c+1, j])
        # = logsumexp_j(log_trans[i,j] + (log_emit[j,c+1] + beta_next[j]))
        inner = log_emit[:, c + 1] + beta_next  # (K,)
        beta_c = logsumexp(log_trans + inner[None, :], axis=1)
        return beta_c, beta_c

    # Scan in reverse: columns C-2, C-3, ..., 0
    _, betas_rev = jax.lax.scan(scan_fn, beta_last, jnp.arange(C - 2, -1, -1))

    # Stack: reversed betas + beta_last
    beta = jnp.concatenate([jnp.flip(betas_rev, axis=0), beta_last[None, :]], axis=0)

    # Log-likelihood from backward: logsumexp(log_init + log_emit[:,0] + beta[0])
    log_likelihood = logsumexp(log_init + log_emit[:, 0] + beta[0])

    return beta, log_likelihood


def hmm_viterbi(cg, tw):
    """Viterbi decoding (scan with max).

    Args:
        cg: CompiledGrammar
        tw: TerminalWeights

    Returns:
        labels: (C,) int32 — best state sequence
        log_prob: scalar — log probability of best path
    """
    log_trans, log_emit, log_init, log_term = _build_hmm_tables(cg, tw)
    K = cg.n_nonterminals
    C = tw.C

    # Viterbi forward
    v_0 = log_init + log_emit[:, 0]

    def scan_fn(v_prev, c):
        # v[c, j] = max_i(v[c-1, i] + log_trans[i, j]) + log_emit[j, c]
        scores = v_prev[:, None] + log_trans  # (K, K)
        v_c = jnp.max(scores, axis=0) + log_emit[:, c]
        bp_c = jnp.argmax(scores, axis=0)  # (K,) backpointer
        return v_c, (v_c, bp_c)

    _, (vs, bps) = jax.lax.scan(scan_fn, v_0, jnp.arange(1, C))

    # Traceback
    # Best final state
    v_final = vs[-1] + log_term if C > 1 else v_0 + log_term
    best_last = jnp.argmax(v_final)
    log_prob = jnp.max(v_final)

    if C == 1:
        return jnp.array([best_last], dtype=jnp.int32), log_prob

    # Reverse scan for traceback
    def traceback_fn(state, bp):
        prev_state = bp[state]
        return prev_state, prev_state

    _, states_rev = jax.lax.scan(traceback_fn, best_last,
                                  jnp.flip(bps, axis=0))
    states = jnp.flip(states_rev)
    labels = jnp.concatenate([states, jnp.array([best_last])]).astype(jnp.int32)

    return labels, log_prob


def hmm_posteriors(cg, tw):
    """Compute posterior state probabilities via forward-backward.

    Args:
        cg: CompiledGrammar
        tw: TerminalWeights

    Returns:
        posteriors: (C, K) — P(state_c = k | x_{1:C})
        log_likelihood: scalar
    """
    alpha, ll_fwd = hmm_forward(cg, tw)
    beta, ll_bwd = hmm_backward(cg, tw)

    # Posterior: P(state_c = k | x) ∝ alpha[c, k] * beta[c, k]
    log_posterior = alpha + beta  # (C, K)
    # Normalize
    log_Z = logsumexp(log_posterior, axis=1, keepdims=True)
    posteriors = jnp.exp(log_posterior - log_Z)

    return posteriors, ll_fwd
