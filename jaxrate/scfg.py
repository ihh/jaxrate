"""SCFG (context-free grammar) algorithms using chart-based DP.

All algorithms operate in log-space for numerical stability.
Complexity: O(C³ * K³) where K = number of nonterminals, C = number of columns.
"""

import jax
import jax.numpy as jnp
from functools import partial

from ._log_semiring import NEG_INF, logsumexp


def _classify_rules(cg, tw):
    """Classify rules for SCFG chart filling.

    Returns:
        terminal_rules: list of (lhs, log_w, model_idx, n_pos) for terminal rules
        unary_rules: list of (lhs, rhs, log_w) for non-emitting unary rules
        emit_unary_rules: list of (lhs, rhs, log_w, model_idx) for left-emitting unary
        binary_rules: list of (lhs, rhs_left, rhs_right, log_w) for binary rules
        emit_paired: list of (lhs, rhs, log_w, model_idx) for paired-emission rules
    """
    terminal_rules = []
    unary_rules = []
    emit_unary_rules = []
    binary_rules = []
    emit_paired = []

    for r in range(cg.n_rules):
        lhs = int(cg.rule_lhs[r])
        n_rhs = int(cg.rule_n_rhs[r])
        log_w = float(cg.rule_log_weights[r])
        n_emit = int(cg.rule_n_emissions[r])

        if n_rhs == 0:
            # Terminal rule: emits column(s)
            if n_emit > 0:
                model_idx = int(cg.rule_emission_model[r, 0])
                n_pos = int(cg.rule_emission_npos[r, 0])
                terminal_rules.append((lhs, log_w, model_idx, n_pos))
        elif n_rhs == 1:
            rhs0 = int(cg.rule_rhs[r, 0])
            if n_emit > 0:
                model_idx = int(cg.rule_emission_model[r, 0])
                n_pos = int(cg.rule_emission_npos[r, 0])
                if n_pos == 2:
                    # Paired emission: A(e₁ x e₂) → B(x)
                    emit_paired.append((lhs, rhs0, log_w, model_idx))
                elif n_pos == 1:
                    # Left-emitting unary: A → e B(x)
                    emit_unary_rules.append((lhs, rhs0, log_w, model_idx))
                else:
                    unary_rules.append((lhs, rhs0, log_w))
            else:
                unary_rules.append((lhs, rhs0, log_w))
        elif n_rhs == 2:
            rhs0 = int(cg.rule_rhs[r, 0])
            rhs1 = int(cg.rule_rhs[r, 1])
            binary_rules.append((lhs, rhs0, rhs1, log_w))

    return terminal_rules, unary_rules, emit_unary_rules, binary_rules, emit_paired


def scfg_inside(cg, tw):
    """Inside algorithm for SCFG.

    Computes alpha[A, i, j] = log P(x_{i:j} | A) for all nonterminals A
    and spans [i, j).

    Args:
        cg: CompiledGrammar (must be 'scfg' class)
        tw: TerminalWeights

    Returns:
        alpha: (K, C, C) log inside probabilities. alpha[A, i, j] is the
               log probability of generating x_{i:j} from nonterminal A.
        log_likelihood: scalar log P(x_{1:C})
    """
    K = cg.n_nonterminals
    C = tw.C

    terminal_rules, unary_rules, emit_unary_rules, binary_rules, emit_paired = \
        _classify_rules(cg, tw)

    # Chart: alpha[A, i, j] for span [i, j) (j exclusive)
    alpha = jnp.full((K, C + 1, C + 1), NEG_INF)

    # Base case: spans of length 1 (terminal emissions)
    for lhs, log_w, model_idx, n_pos in terminal_rules:
        if n_pos == 1:
            # Single emission at column i: span [i, i+1)
            for i in range(C):
                emit_w = tw.single[model_idx, i]
                alpha = alpha.at[lhs, i, i + 1].set(
                    jnp.logaddexp(alpha[lhs, i, i + 1], log_w + emit_w))

    # Fill chart bottom-up by span length
    for span_len in range(2, C + 1):
        for i in range(C - span_len + 1):
            j = i + span_len

            # Left-emitting unary rules: A → e B(x), span [i, j) = emit i + B[i+1, j)
            for lhs, rhs, log_w, model_idx in emit_unary_rules:
                emit_w = tw.single[model_idx, i]
                inner = alpha[rhs, i + 1, j]
                score = log_w + emit_w + inner
                alpha = alpha.at[lhs, i, j].set(
                    jnp.logaddexp(alpha[lhs, i, j], score))

            # Paired emission rules: A → e₁ B e₂
            for lhs, rhs, log_w, model_idx in emit_paired:
                if tw.paired is not None and span_len >= 2:
                    emit_w = tw.paired[model_idx, i, j - 1]
                    inner = alpha[rhs, i + 1, j - 1]
                    score = log_w + emit_w + inner
                    alpha = alpha.at[lhs, i, j].set(
                        jnp.logaddexp(alpha[lhs, i, j], score))

            # Binary rules: A → B C
            for lhs, rhs0, rhs1, log_w in binary_rules:
                for k in range(i + 1, j):
                    score = log_w + alpha[rhs0, i, k] + alpha[rhs1, k, j]
                    alpha = alpha.at[lhs, i, j].set(
                        jnp.logaddexp(alpha[lhs, i, j], score))

            # Non-emitting unary rules: A → B
            for lhs, rhs, log_w in unary_rules:
                score = log_w + alpha[rhs, i, j]
                alpha = alpha.at[lhs, i, j].set(
                    jnp.logaddexp(alpha[lhs, i, j], score))

    log_likelihood = alpha[cg.start, 0, C]
    return alpha, log_likelihood


def scfg_outside(cg, tw, alpha):
    """Outside algorithm for SCFG.

    Computes beta[A, i, j] = log P(x_{1:i}, x_{j:C} | A generates x_{i:j}).

    Args:
        cg: CompiledGrammar
        tw: TerminalWeights
        alpha: (K, C+1, C+1) inside chart from scfg_inside

    Returns:
        beta: (K, C+1, C+1) log outside probabilities
    """
    K = cg.n_nonterminals
    C = tw.C

    terminal_rules, unary_rules, emit_unary_rules, binary_rules, emit_paired = \
        _classify_rules(cg, tw)

    beta = jnp.full((K, C + 1, C + 1), NEG_INF)
    # Start symbol spans the whole sequence
    beta = beta.at[cg.start, 0, C].set(0.0)

    # Fill top-down by decreasing span length
    for span_len in range(C, 0, -1):
        for i in range(C - span_len + 1):
            j = i + span_len

            # Non-emitting unary rules: A → B (parent A, child B)
            for lhs, rhs, log_w in unary_rules:
                score = beta[lhs, i, j] + log_w
                beta = beta.at[rhs, i, j].set(
                    jnp.logaddexp(beta[rhs, i, j], score))

            # Left-emitting unary: A → e B, outside for B[i+1, j)
            for lhs, rhs, log_w, model_idx in emit_unary_rules:
                if span_len >= 2:
                    emit_w = tw.single[model_idx, i]
                    score = beta[lhs, i, j] + log_w + emit_w
                    beta = beta.at[rhs, i + 1, j].set(
                        jnp.logaddexp(beta[rhs, i + 1, j], score))

            # Binary rules: A → B C
            for lhs, rhs0, rhs1, log_w in binary_rules:
                # Outside for B[i, k]: parent A[i, j], sibling C[k, j]
                for k in range(i + 1, j):
                    score_b = beta[lhs, i, j] + log_w + alpha[rhs1, k, j]
                    beta = beta.at[rhs0, i, k].set(
                        jnp.logaddexp(beta[rhs0, i, k], score_b))

                    score_c = beta[lhs, i, j] + log_w + alpha[rhs0, i, k]
                    beta = beta.at[rhs1, k, j].set(
                        jnp.logaddexp(beta[rhs1, k, j], score_c))

            # Paired emission: A → e₁ B e₂
            for lhs, rhs, log_w, model_idx in emit_paired:
                if tw.paired is not None and span_len >= 2:
                    emit_w = tw.paired[model_idx, i, j - 1]
                    score = beta[lhs, i, j] + log_w + emit_w
                    beta = beta.at[rhs, i + 1, j - 1].set(
                        jnp.logaddexp(beta[rhs, i + 1, j - 1], score))

    return beta


def scfg_viterbi(cg, tw):
    """CYK/Viterbi decoding for SCFG.

    Args:
        cg: CompiledGrammar
        tw: TerminalWeights

    Returns:
        labels: (C,) int32 — per-column nonterminal labels from best parse
        log_prob: scalar — log probability of best parse
        backpointers: dict of backpointer info for tree reconstruction
    """
    K = cg.n_nonterminals
    C = tw.C

    terminal_rules, unary_rules, emit_unary_rules, binary_rules, emit_paired = \
        _classify_rules(cg, tw)

    # Viterbi chart (max instead of logaddexp)
    v = jnp.full((K, C + 1, C + 1), NEG_INF)
    bp = {}  # backpointers: (A, i, j) → rule info

    # Base case
    for lhs, log_w, model_idx, n_pos in terminal_rules:
        if n_pos == 1:
            for i in range(C):
                score = log_w + tw.single[model_idx, i]
                if score > v[lhs, i, i + 1]:
                    v = v.at[lhs, i, i + 1].set(score)
                    bp[(lhs, i, i + 1)] = ('terminal', model_idx)

    # Fill chart
    for span_len in range(2, C + 1):
        for i in range(C - span_len + 1):
            j = i + span_len

            # Left-emitting unary: A → e B
            for lhs, rhs, log_w, model_idx in emit_unary_rules:
                emit_w = float(tw.single[model_idx, i])
                inner = float(v[rhs, i + 1, j])
                score = log_w + emit_w + inner
                if score > float(v[lhs, i, j]):
                    v = v.at[lhs, i, j].set(score)
                    bp[(lhs, i, j)] = ('emit_unary', rhs, i + 1, j, model_idx)

            for lhs, rhs, log_w, model_idx in emit_paired:
                if tw.paired is not None and span_len >= 2:
                    emit_w = float(tw.paired[model_idx, i, j - 1])
                    inner = float(v[rhs, i + 1, j - 1])
                    score = log_w + emit_w + inner
                    if score > float(v[lhs, i, j]):
                        v = v.at[lhs, i, j].set(score)
                        bp[(lhs, i, j)] = ('paired', rhs, i + 1, j - 1, model_idx)

            for lhs, rhs0, rhs1, log_w in binary_rules:
                for k in range(i + 1, j):
                    score = log_w + float(v[rhs0, i, k]) + float(v[rhs1, k, j])
                    if score > float(v[lhs, i, j]):
                        v = v.at[lhs, i, j].set(score)
                        bp[(lhs, i, j)] = ('binary', rhs0, rhs1, k)

            for lhs, rhs, log_w in unary_rules:
                score = log_w + float(v[rhs, i, j])
                if score > float(v[lhs, i, j]):
                    v = v.at[lhs, i, j].set(score)
                    bp[(lhs, i, j)] = ('unary', rhs)

    log_prob = float(v[cg.start, 0, C])

    # Extract per-column labels from backpointers
    labels = jnp.zeros(C, dtype=jnp.int32)

    def _trace(A, i, j):
        nonlocal labels
        if i >= j:
            return
        key = (A, i, j)
        if key not in bp:
            return
        info = bp[key]
        if info[0] == 'terminal':
            labels = labels.at[i].set(A)
        elif info[0] == 'emit_unary':
            _, rhs, ci, cj, _ = info
            labels = labels.at[i].set(A)
            _trace(rhs, ci, cj)
        elif info[0] == 'paired':
            _, rhs, ci, cj, _ = info
            labels = labels.at[i].set(A)
            labels = labels.at[j - 1].set(A)
            _trace(rhs, ci, cj)
        elif info[0] == 'binary':
            _, rhs0, rhs1, k = info
            _trace(rhs0, i, k)
            _trace(rhs1, k, j)
        elif info[0] == 'unary':
            _, rhs = info
            _trace(rhs, i, j)

    _trace(cg.start, 0, C)

    return labels, log_prob, bp
