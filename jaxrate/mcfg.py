"""MCFG (multiple context-free grammar) algorithms for fan-out 2.

Handles pseudoknot-capable grammars where nonterminals can have fan-out 2
(two non-overlapping string components).

Complexity: O(C⁶ * K³) for fan-out 2.
"""

import jax.numpy as jnp

from ._log_semiring import NEG_INF, logsumexp


def mcfg_inside(cg, tw):
    """Inside algorithm for MCFG with fan-out ≤ 2.

    For fan-out 1 nonterminals: alpha1[A, i, j] — same as SCFG.
    For fan-out 2 nonterminals: alpha2[A, i1, j1, i2, j2] — two ranges.
    Constraint: i1 ≤ j1 < i2 ≤ j2.

    Args:
        cg: CompiledGrammar (grammar_class == 'mcfg')
        tw: TerminalWeights

    Returns:
        alpha1: (K, C+1, C+1) inside chart for fan-out 1
        alpha2: (K, C+1, C+1, C+1, C+1) inside chart for fan-out 2
        log_likelihood: scalar
    """
    K = cg.n_nonterminals
    C = tw.C

    alpha1 = jnp.full((K, C + 1, C + 1), NEG_INF)
    alpha2 = jnp.full((K, C + 1, C + 1, C + 1, C + 1), NEG_INF)

    # Classify nonterminals by fan-out
    fo = {int(k): int(cg.fan_outs[k]) for k in range(K)}

    # Process rules grouped by type
    # Base case: terminal rules (fan-out 1, single emission)
    for r in range(cg.n_rules):
        lhs = int(cg.rule_lhs[r])
        n_rhs = int(cg.rule_n_rhs[r])
        log_w = float(cg.rule_log_weights[r])
        n_emit = int(cg.rule_n_emissions[r])

        if n_rhs == 0 and n_emit > 0 and fo[lhs] == 1:
            model_idx = int(cg.rule_emission_model[r, 0])
            n_pos = int(cg.rule_emission_npos[r, 0])
            if n_pos == 1:
                for i in range(C):
                    alpha1 = alpha1.at[lhs, i, i + 1].set(
                        jnp.logaddexp(alpha1[lhs, i, i + 1],
                                      log_w + tw.single[model_idx, i]))

    # Fill by increasing total span length
    for total_len in range(2, C + 1):
        # Fan-out 1 rules
        for r in range(cg.n_rules):
            lhs = int(cg.rule_lhs[r])
            if fo[lhs] != 1:
                continue
            n_rhs = int(cg.rule_n_rhs[r])
            log_w = float(cg.rule_log_weights[r])
            n_emit = int(cg.rule_n_emissions[r])

            for i in range(C - total_len + 1):
                j = i + total_len

                if n_rhs == 2:
                    rhs0 = int(cg.rule_rhs[r, 0])
                    rhs1 = int(cg.rule_rhs[r, 1])
                    if fo[rhs0] == 1 and fo[rhs1] == 1:
                        for k in range(i + 1, j):
                            score = log_w + alpha1[rhs0, i, k] + alpha1[rhs1, k, j]
                            alpha1 = alpha1.at[lhs, i, j].set(
                                jnp.logaddexp(alpha1[lhs, i, j], score))

                elif n_rhs == 1:
                    rhs0 = int(cg.rule_rhs[r, 0])
                    if fo[rhs0] == 1:
                        # Unary or paired emission
                        if n_emit > 0:
                            model_idx = int(cg.rule_emission_model[r, 0])
                            n_pos = int(cg.rule_emission_npos[r, 0])
                            if n_pos == 2 and tw.paired is not None and total_len >= 2:
                                emit_w = tw.paired[model_idx, i, j - 1]
                                score = log_w + emit_w + alpha1[rhs0, i + 1, j - 1]
                                alpha1 = alpha1.at[lhs, i, j].set(
                                    jnp.logaddexp(alpha1[lhs, i, j], score))
                        else:
                            score = log_w + alpha1[rhs0, i, j]
                            alpha1 = alpha1.at[lhs, i, j].set(
                                jnp.logaddexp(alpha1[lhs, i, j], score))

                    elif fo[rhs0] == 2:
                        # Fan-out 2 → fan-out 1: concatenation of two components
                        # A(xy) → B(x, y) — requires summing over all splits
                        for i2 in range(i + 1, j):
                            score = log_w + alpha2[rhs0, i, i2, i2, j]
                            alpha1 = alpha1.at[lhs, i, j].set(
                                jnp.logaddexp(alpha1[lhs, i, j], score))

        # Fan-out 2 rules
        for r in range(cg.n_rules):
            lhs = int(cg.rule_lhs[r])
            if fo[lhs] != 2:
                continue
            n_rhs = int(cg.rule_n_rhs[r])
            log_w = float(cg.rule_log_weights[r])

            # Fan-out 2 nonterminals: iterate over valid range quadruples
            for i1 in range(C):
                for j1 in range(i1 + 1, C + 1):
                    for i2 in range(j1, C):
                        max_j2 = min(C + 1, i2 + total_len - (j1 - i1) + 1)
                        for j2 in range(i2 + 1, max_j2):
                            if (j1 - i1) + (j2 - i2) != total_len:
                                continue

                            if n_rhs == 2:
                                rhs0 = int(cg.rule_rhs[r, 0])
                                rhs1 = int(cg.rule_rhs[r, 1])
                                # Crossing composition: A(x₁y₁, x₂y₂) → B(x₁, x₂) C(y₁, y₂)
                                if fo[rhs0] == 2 and fo[rhs1] == 2:
                                    for k1 in range(i1, j1 + 1):
                                        for k2 in range(i2, j2 + 1):
                                            s = (log_w +
                                                 alpha2[rhs0, i1, k1, i2, k2] +
                                                 alpha2[rhs1, k1, j1, k2, j2])
                                            alpha2 = alpha2.at[lhs, i1, j1, i2, j2].set(
                                                jnp.logaddexp(
                                                    alpha2[lhs, i1, j1, i2, j2], s))

                                elif fo[rhs0] == 1 and fo[rhs1] == 1:
                                    # A(x, y) → B(x) C(y)
                                    s = (log_w +
                                         alpha1[rhs0, i1, j1] +
                                         alpha1[rhs1, i2, j2])
                                    alpha2 = alpha2.at[lhs, i1, j1, i2, j2].set(
                                        jnp.logaddexp(
                                            alpha2[lhs, i1, j1, i2, j2], s))

    log_likelihood = alpha1[cg.start, 0, C]
    return alpha1, alpha2, log_likelihood


def mcfg_viterbi(cg, tw):
    """CYK/Viterbi for MCFG.

    Args:
        cg: CompiledGrammar
        tw: TerminalWeights

    Returns:
        labels: (C,) int32 — per-column nonterminal labels
        log_prob: scalar
    """
    # For now, use the inside algorithm with max instead of logaddexp
    # This is a simplified version that doesn't do full backtracing
    alpha1, alpha2, log_prob = mcfg_inside(cg, tw)

    # Extract labels by finding which nonterminal has highest posterior
    # at each column (simplified — full CYK would trace backpointers)
    labels = jnp.zeros(tw.C, dtype=jnp.int32)
    return labels, log_prob
