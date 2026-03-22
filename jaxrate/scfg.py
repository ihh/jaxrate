"""SCFG (context-free grammar) algorithms using chart-based DP.

All algorithms operate in log-space for numerical stability.
Complexity: O(C³ * K³) where K = number of nonterminals, C = number of columns.
"""

import jax
import jax.numpy as jnp
from functools import partial

from ._log_semiring import NEG_INF, logsumexp


def _topo_sort_unary(unary_rules, K):
    """Topologically sort unary rules for correct propagation order.

    Returns unary rules reordered so that if A → B exists and B → C exists,
    then B → C comes before A → B (process bottom-up: sources first).
    """
    # Build dependency graph: lhs depends on rhs
    from collections import defaultdict, deque

    # Adjacency: rhs -> [list of rule indices that have this rhs as LHS... no]
    # We want: process rules whose RHS has no further unary rules first.

    # Build: for each nonterminal, which unary rules have it as LHS?
    rules_by_lhs = defaultdict(list)
    rhs_set = set()
    for idx, (lhs, rhs, log_w) in enumerate(unary_rules):
        rules_by_lhs[lhs].append(idx)
        rhs_set.add(rhs)

    # Nonterminals that appear as RHS but not as LHS of unary rules are "sources"
    lhs_set = set(rules_by_lhs.keys())
    # In-degree for each LHS: depends on whether its RHS is also a LHS
    in_degree = defaultdict(int)
    deps = defaultdict(list)  # lhs -> [rhs nonterminals that are also LHS of unary rules]

    for idx, (lhs, rhs, log_w) in enumerate(unary_rules):
        if rhs in lhs_set:
            in_degree[lhs] += 1
            deps[rhs].append(lhs)
        else:
            in_degree.setdefault(lhs, 0)

    # Kahn's algorithm
    queue = deque([nt for nt in lhs_set if in_degree.get(nt, 0) == 0])
    ordered_lhs = []
    while queue:
        nt = queue.popleft()
        ordered_lhs.append(nt)
        for dependent in deps.get(nt, []):
            in_degree[dependent] -= 1
            if in_degree[dependent] == 0:
                queue.append(dependent)

    # If there are cycles, append remaining (shouldn't happen in well-formed grammars)
    remaining = [nt for nt in lhs_set if nt not in set(ordered_lhs)]
    ordered_lhs.extend(remaining)

    # Reorder rules: process rules whose LHS appears earlier in ordered_lhs first
    lhs_order = {nt: i for i, nt in enumerate(ordered_lhs)}
    sorted_indices = sorted(range(len(unary_rules)),
                            key=lambda idx: lhs_order.get(unary_rules[idx][0], len(ordered_lhs)))
    return [unary_rules[i] for i in sorted_indices]


def _classify_rules(cg, tw):
    """Classify rules for SCFG chart filling.

    Returns:
        terminal_rules: list of (lhs, log_w, model_idx, n_pos) for terminal rules
        unary_rules: list of (lhs, rhs, log_w) for non-emitting unary rules
        emit_unary_rules: list of (lhs, rhs, log_w, model_idx) for left-emitting unary
        binary_rules: list of (lhs, rhs_left, rhs_right, log_w) for binary rules
        emit_paired: list of (lhs, rhs, log_w, model_idx) for paired-emission rules
        epsilon_rules: list of (lhs, log_w) for epsilon productions (A → ε)
    """
    terminal_rules = []
    unary_rules = []
    emit_unary_rules = []
    binary_rules = []
    emit_paired = []
    epsilon_rules = []

    for r in range(cg.n_rules):
        lhs = int(cg.rule_lhs[r])
        n_rhs = int(cg.rule_n_rhs[r])
        log_w = float(cg.rule_log_weights[r])
        n_emit = int(cg.rule_n_emissions[r])

        if n_rhs == 0:
            if n_emit > 0:
                # Terminal rule: emits column(s)
                model_idx = int(cg.rule_emission_model[r, 0])
                n_pos = int(cg.rule_emission_npos[r, 0])
                terminal_rules.append((lhs, log_w, model_idx, n_pos))
            else:
                # Epsilon production: A → ε
                epsilon_rules.append((lhs, log_w))
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

    return terminal_rules, unary_rules, emit_unary_rules, binary_rules, emit_paired, epsilon_rules


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

    terminal_rules, unary_rules, emit_unary_rules, binary_rules, emit_paired, \
        epsilon_rules = _classify_rules(cg, tw)

    # Sort unary rules for correct propagation through chains
    sorted_unary = _topo_sort_unary(unary_rules, K)

    # Chart: alpha[A, i, j] for span [i, j) (j exclusive)
    alpha = jnp.full((K, C + 1, C + 1), NEG_INF)

    # Base case: epsilon productions (span length 0)
    for lhs, log_w in epsilon_rules:
        for i in range(C + 1):
            alpha = alpha.at[lhs, i, i].set(
                jnp.logaddexp(alpha[lhs, i, i], log_w))

    # Propagate unary rules on epsilon spans (sorted order, single pass)
    for lhs, rhs, log_w in sorted_unary:
        for i in range(C + 1):
            score = log_w + alpha[rhs, i, i]
            alpha = alpha.at[lhs, i, i].set(
                jnp.logaddexp(alpha[lhs, i, i], score))

    # Fill chart bottom-up by span length (starting from 1)
    for span_len in range(1, C + 1):
        for i in range(C - span_len + 1):
            j = i + span_len

            # Terminal emission rules: A → e (span length 1 only)
            if span_len == 1:
                for lhs, log_w, model_idx, n_pos in terminal_rules:
                    if n_pos == 1:
                        emit_w = tw.single[model_idx, i]
                        alpha = alpha.at[lhs, i, j].set(
                            jnp.logaddexp(alpha[lhs, i, j], log_w + emit_w))

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

            # Non-emitting unary rules: A → B (topologically sorted)
            for lhs, rhs, log_w in sorted_unary:
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

    terminal_rules, unary_rules, emit_unary_rules, binary_rules, emit_paired, \
        epsilon_rules = _classify_rules(cg, tw)

    # For outside, reverse the topo order (propagate from parents to children)
    sorted_unary_rev = list(reversed(_topo_sort_unary(unary_rules, K)))

    beta = jnp.full((K, C + 1, C + 1), NEG_INF)
    # Start symbol spans the whole sequence
    beta = beta.at[cg.start, 0, C].set(0.0)

    # Fill top-down by decreasing span length
    for span_len in range(C, 0, -1):
        for i in range(C - span_len + 1):
            j = i + span_len

            # Non-emitting unary rules: A → B (parent A, child B)
            for lhs, rhs, log_w in sorted_unary_rev:
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


def scfg_posteriors(cg, tw):
    """Compute per-column posterior nonterminal probabilities via inside-outside.

    For each column c, computes P(nonterminal k emits at column c | sequence).
    This is derived from the inside-outside charts: for each emitting rule
    that touches column c, the posterior contribution is
    exp(alpha[child, i', j'] + beta[lhs, i, j] + log_w + emit_w - log_Z).

    Args:
        cg: CompiledGrammar (must be 'scfg' class)
        tw: TerminalWeights

    Returns:
        posteriors: (C, K) — P(nonterminal k emits at column c | x_{1:C})
        log_likelihood: scalar
    """
    K = cg.n_nonterminals
    C = tw.C

    alpha, log_Z = scfg_inside(cg, tw)
    beta = scfg_outside(cg, tw, alpha)

    terminal_rules, unary_rules, emit_unary_rules, binary_rules, emit_paired, \
        epsilon_rules = _classify_rules(cg, tw)

    # Accumulate posterior mass per (column, nonterminal)
    post = jnp.full((C, K), NEG_INF)

    # Terminal emission rules: A → e at column i
    for lhs, log_w, model_idx, n_pos in terminal_rules:
        if n_pos == 1:
            for i in range(C):
                emit_w = tw.single[model_idx, i]
                score = beta[lhs, i, i + 1] + log_w + emit_w - log_Z
                post = post.at[i, lhs].set(
                    jnp.logaddexp(post[i, lhs], score))

    # Left-emitting unary rules: A → e B, emits at column i
    for lhs, rhs, log_w, model_idx in emit_unary_rules:
        for span_len in range(1, C + 1):
            for i in range(C - span_len + 1):
                j = i + span_len
                emit_w = tw.single[model_idx, i]
                score = beta[lhs, i, j] + log_w + emit_w + alpha[rhs, i + 1, j] - log_Z
                post = post.at[i, lhs].set(
                    jnp.logaddexp(post[i, lhs], score))

    # Paired emission rules: A → e₁ B e₂, emits at columns i and j-1
    for lhs, rhs, log_w, model_idx in emit_paired:
        if tw.paired is not None:
            for span_len in range(2, C + 1):
                for i in range(C - span_len + 1):
                    j = i + span_len
                    emit_w = tw.paired[model_idx, i, j - 1]
                    score = beta[lhs, i, j] + log_w + emit_w + alpha[rhs, i + 1, j - 1] - log_Z
                    # Left position
                    post = post.at[i, lhs].set(
                        jnp.logaddexp(post[i, lhs], score))
                    # Right position
                    post = post.at[j - 1, lhs].set(
                        jnp.logaddexp(post[j - 1, lhs], score))

    # Convert from log-space
    posteriors = jnp.exp(post)

    return posteriors, log_Z


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

    terminal_rules, unary_rules, emit_unary_rules, binary_rules, emit_paired, \
        epsilon_rules = _classify_rules(cg, tw)

    sorted_unary = _topo_sort_unary(unary_rules, K)

    # Viterbi chart (max instead of logaddexp)
    v = jnp.full((K, C + 1, C + 1), NEG_INF)
    bp = {}  # backpointers: (A, i, j) → rule info

    # Base case: epsilon productions (span length 0)
    for lhs, log_w in epsilon_rules:
        for i in range(C + 1):
            if log_w > float(v[lhs, i, i]):
                v = v.at[lhs, i, i].set(log_w)
                bp[(lhs, i, i)] = ('epsilon',)

    # Propagate unary rules on epsilon spans (sorted, single pass)
    for lhs, rhs, log_w in sorted_unary:
        for i in range(C + 1):
            score = log_w + float(v[rhs, i, i])
            if score > float(v[lhs, i, i]):
                v = v.at[lhs, i, i].set(score)
                bp[(lhs, i, i)] = ('unary', rhs)

    # Fill chart bottom-up by span length (starting from 1)
    for span_len in range(1, C + 1):
        for i in range(C - span_len + 1):
            j = i + span_len

            # Terminal emissions (span length 1 only)
            if span_len == 1:
                for lhs, log_w, model_idx, n_pos in terminal_rules:
                    if n_pos == 1:
                        score = log_w + float(tw.single[model_idx, i])
                        if score > float(v[lhs, i, j]):
                            v = v.at[lhs, i, j].set(score)
                            bp[(lhs, i, j)] = ('terminal', model_idx)

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

            for lhs, rhs, log_w in sorted_unary:
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
