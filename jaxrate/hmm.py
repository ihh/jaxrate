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

    Handles both flat right-linear rules (LHS → emit RHS) and the xrate
    two-step convention (S → emit S*, S* → RHS) by eliminating null
    (non-emitting) states.

    Returns tables indexed by the FULL nonterminal set (K = n_nonterminals).
    Null states get NEG_INF emission weights so they contribute nothing
    to the forward/backward computation.

    We build:
    - log_trans: (K, K) log transition matrix (trans[i,j] = log P(i → j))
    - log_emit: (K, C) log emission weights per state per column
    - log_init: (K,) initial state log-probabilities
    - log_term: (K,) log-probability of terminating at each state
    """
    import numpy as np

    K = cg.n_nonterminals
    C = tw.C

    # ------------------------------------------------------------------
    # Step 1: Classify rules and build raw transition/emit/term tables
    # ------------------------------------------------------------------
    # Raw tables over ALL nonterminals (including null states)
    raw_log_trans = np.full((K, K), NEG_INF)
    raw_log_term = np.full(K, NEG_INF)
    emit_model_for = np.full(K, -1, dtype=np.int32)
    emit_npos_for = np.ones(K, dtype=np.int32)
    has_emit = np.zeros(K, dtype=bool)

    for r in range(cg.n_rules):
        lhs = int(cg.rule_lhs[r])
        n_rhs = int(cg.rule_n_rhs[r])
        log_w = float(cg.rule_log_weights[r])
        n_emit = int(cg.rule_n_emissions[r])

        if n_emit > 0:
            emit_model_for[lhs] = int(cg.rule_emission_model[r, 0])
            emit_npos_for[lhs] = int(cg.rule_emission_npos[r, 0])
            has_emit[lhs] = True

        if n_rhs == 0:
            raw_log_term[lhs] = float(
                jnp.logaddexp(raw_log_term[lhs], log_w))
        else:
            rhs0 = int(cg.rule_rhs[r, 0])
            raw_log_trans[lhs, rhs0] = float(
                jnp.logaddexp(raw_log_trans[lhs, rhs0], log_w))

    # ------------------------------------------------------------------
    # Step 2: Eliminate null (non-emitting) states
    # ------------------------------------------------------------------
    # A null state is one that never emits in any of its rules.
    # For the xrate pattern (S → emit S*, S* → {transitions}), S* is null.
    # We compose transitions through null chains: if path is
    #   emitting_i → null_a → null_b → emitting_j
    # the effective transition prob is product of the link probs.
    #
    # In log space: iterate log_trans through null states until convergence.

    is_null = ~has_emit

    if np.any(is_null):
        # Propagate transitions through null states using log-space
        # matrix "multiplication" (logsumexp over intermediate null states).
        # Iterate until no null→null transitions remain.
        lt = raw_log_trans.copy()
        lterm = raw_log_term.copy()
        null_idx = np.where(is_null)[0]

        for _iteration in range(K + 1):
            changed = False
            for n in null_idx:
                # For every state i that transitions to null state n:
                for i in range(K):
                    if lt[i, n] <= NEG_INF + 1:
                        continue
                    log_p_i_n = lt[i, n]
                    # Compose: i → n → j  becomes i → j
                    for j in range(K):
                        if lt[n, j] <= NEG_INF + 1:
                            continue
                        new_val = log_p_i_n + lt[n, j]
                        old_val = lt[i, j]
                        combined = float(jnp.logaddexp(old_val, new_val))
                        if combined > old_val + 1e-12:
                            lt[i, j] = combined
                            changed = True
                    # Compose termination: i → n → end
                    if lterm[n] > NEG_INF + 1:
                        new_term = log_p_i_n + lterm[n]
                        old_term = lterm[i]
                        combined = float(jnp.logaddexp(old_term, new_term))
                        if combined > old_term + 1e-12:
                            lterm[i] = combined
                            changed = True
                    # Remove the i → n transition (it's been composed)
                    lt[i, n] = NEG_INF
            if not changed:
                break

        raw_log_trans = lt
        raw_log_term = lterm

    # ------------------------------------------------------------------
    # Step 3: Build JAX arrays
    # ------------------------------------------------------------------
    log_trans = jnp.array(raw_log_trans)
    log_term = jnp.array(raw_log_term)

    # Emission table: (K, C)
    log_emit = jnp.full((K, C), NEG_INF)
    for k in range(K):
        model_idx = int(emit_model_for[k])
        if model_idx >= 0:
            npos = int(emit_npos_for[k])
            if npos == 1:
                log_emit = log_emit.at[k].set(tw.single[model_idx])
            elif tw.kmer is not None:
                log_emit = log_emit.at[k].set(
                    jnp.pad(tw.kmer[model_idx],
                            (0, max(0, C - tw.kmer[model_idx].shape[0])),
                            constant_values=NEG_INF)[:C])
        # Null states keep NEG_INF emissions — they don't participate

    # Initial distribution: propagate through null start state
    log_init = jnp.full((K,), NEG_INF)
    if is_null[cg.start]:
        # Start is null — distribute to its successors
        for j in range(K):
            if raw_log_trans[cg.start, j] > NEG_INF + 1:
                log_init = log_init.at[j].set(
                    jnp.logaddexp(log_init[j], raw_log_trans[cg.start, j]))
    else:
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
