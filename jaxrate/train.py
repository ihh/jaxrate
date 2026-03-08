"""EM training and gradient-based optimization for grammar parameters."""

import jax
import jax.numpy as jnp
import numpy as np

from .types import TrainState, Grammar, Rule
from .grammar import compile_grammar
from .inside import inside
from .outside import outside
from ._log_semiring import logsumexp, NEG_INF


def _compute_expected_counts_hmm(cg, tw, alpha, beta, ll):
    """Compute expected rule counts for HMM via forward-backward."""
    from .hmm import _build_hmm_tables
    log_trans, log_emit, log_init, log_term = _build_hmm_tables(cg, tw)
    K = cg.n_nonterminals
    C = tw.C

    # Expected transition counts
    trans_counts = jnp.zeros((K, K))
    for c in range(C - 1):
        # P(state_c=i, state_{c+1}=j | x) ∝ alpha[c,i] * trans[i,j] * emit[j,c+1] * beta[c+1,j]
        log_joint = (alpha[c, :, None] + log_trans +
                     log_emit[None, :, c + 1] + beta[c + 1, None, :])
        trans_counts = trans_counts + jnp.exp(log_joint - ll)

    return trans_counts


def em_step(grammar, terminal_weights):
    """One E-step + M-step of EM training.

    E-step: inside-outside → expected rule usage counts.
    M-step: normalize counts per LHS nonterminal → new log-weights.

    Args:
        grammar: Grammar
        terminal_weights: TerminalWeights

    Returns:
        new_grammar: Grammar with updated rule log-weights
        log_likelihood: scalar log P(x | grammar)
    """
    cg = compile_grammar(grammar)
    chart, ll = inside(cg, terminal_weights)

    # For HMM, use forward-backward expected counts
    if cg.grammar_class == 'hmm':
        beta_chart, _ = outside(cg, terminal_weights)
        trans_counts = _compute_expected_counts_hmm(cg, terminal_weights,
                                                     chart, beta_chart, ll)
        # Map transition counts back to rule log-weights
        new_rules = []
        for rule in grammar.rules:
            new_rules.append(rule)  # Placeholder — real impl maps counts to rules

        # For now, normalize per-LHS
        n_nt = len(grammar.nonterminals)
        rule_counts = jnp.ones(len(grammar.rules))  # placeholder

        # Group by LHS and normalize
        new_log_weights = jnp.zeros(len(grammar.rules))
        for nt_idx in range(n_nt):
            mask = jnp.array([r.lhs == nt_idx for r in grammar.rules])
            if jnp.any(mask):
                nt_counts = jnp.where(mask, rule_counts, 0.0)
                total = jnp.sum(nt_counts)
                nt_log_weights = jnp.where(
                    mask, jnp.log(nt_counts + 1e-30) - jnp.log(total + 1e-30),
                    NEG_INF)
                new_log_weights = jnp.where(mask, nt_log_weights, new_log_weights)

        new_rules = [
            Rule(r.lhs, r.rhs, r.emissions, float(new_log_weights[i]),
                 r.composition)
            for i, r in enumerate(grammar.rules)
        ]
    else:
        # SCFG/MCFG: use inside-outside expected counts
        beta_chart, _ = outside(cg, terminal_weights, inside_chart=chart)
        # Simplified: just return same rules for now
        new_rules = list(grammar.rules)

    new_grammar = Grammar(
        nonterminals=grammar.nonterminals,
        rules=new_rules,
        start=grammar.start,
        n_models=grammar.n_models,
    )

    return new_grammar, float(ll)


def train(grammar, terminal_weights, n_iterations=100, convergence_tol=1e-6):
    """Run EM training loop.

    Args:
        grammar: Grammar
        terminal_weights: TerminalWeights
        n_iterations: maximum iterations
        convergence_tol: stop when |ΔLL| < tol

    Returns:
        trained_grammar: Grammar with optimized log-weights
        train_state: TrainState with final state
        history: list of (iteration, log_likelihood) tuples
    """
    current_grammar = grammar
    prev_ll = float('-inf')
    history = []

    for iteration in range(n_iterations):
        current_grammar, ll = em_step(current_grammar, terminal_weights)
        history.append((iteration, ll))

        if abs(ll - prev_ll) < convergence_tol and iteration > 0:
            break
        prev_ll = ll

    cg = compile_grammar(current_grammar)
    log_weights = jnp.array([r.log_weight for r in current_grammar.rules])
    state = TrainState(
        log_weights=log_weights,
        iteration=len(history),
        log_likelihood=prev_ll,
    )

    return current_grammar, state, history
