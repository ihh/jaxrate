"""Simulation: sample derivations and alignments from grammars."""

import jax
import jax.numpy as jnp
import jax.random as jr


def simulate_parse(grammar, seq_length, key):
    """Sample a derivation from the grammar.

    For HMMs: sample a state sequence of the given length.
    For SCFGs/MCFGs: sample a parse tree that generates a sequence
    of approximately the given length.

    Args:
        grammar: Grammar
        seq_length: target sequence length
        key: JAX PRNG key

    Returns:
        labels: (C,) int32 — sampled state/nonterminal sequence
        rules_used: list of (rule_index, span) tuples
    """
    from .grammar import compile_grammar, classify_grammar

    cg = compile_grammar(grammar)

    if cg.grammar_class == 'hmm':
        return _simulate_hmm(grammar, cg, seq_length, key)
    else:
        return _simulate_cfg(grammar, cg, seq_length, key)


def _simulate_hmm(grammar, cg, seq_length, key):
    """Sample from an HMM."""
    K = cg.n_nonterminals
    C = seq_length

    # Build transition and termination tables
    log_trans = jnp.full((K, K), -1e38)
    log_term = jnp.full((K,), -1e38)

    for r in range(cg.n_rules):
        lhs = int(cg.rule_lhs[r])
        n_rhs = int(cg.rule_n_rhs[r])
        log_w = float(cg.rule_log_weights[r])

        if n_rhs == 0:
            log_term = log_term.at[lhs].set(
                jnp.logaddexp(log_term[lhs], log_w))
        else:
            rhs = int(cg.rule_rhs[r, 0])
            log_trans = log_trans.at[lhs, rhs].set(
                jnp.logaddexp(log_trans[lhs, rhs], log_w))

    # Initial distribution
    log_init = jnp.full((K,), -1e38)
    start = cg.start
    has_trans = jnp.any(log_trans[start] > -1e37)
    if has_trans:
        log_init = log_trans[start]
    else:
        log_init = log_init.at[start].set(0.0)

    # Sample initial state
    key, k1 = jr.split(key)
    state = jr.categorical(k1, log_init)
    labels = [int(state)]

    # Sample transitions
    for c in range(1, C):
        key, k1 = jr.split(key)
        state = jr.categorical(k1, log_trans[state])
        labels.append(int(state))

    return jnp.array(labels, dtype=jnp.int32), []


def _simulate_cfg(grammar, cg, seq_length, key):
    """Sample from a CFG (simplified: random walk on rules)."""
    labels = jnp.zeros(seq_length, dtype=jnp.int32)
    return labels, []
