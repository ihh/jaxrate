"""MCFG (multiple context-free grammar) algorithms for fan-out 2.

Handles pseudoknot-capable grammars where nonterminals can have fan-out 2
(two non-overlapping string components).

Complexity: O(C⁶ * K³) for fan-out 2, or O(C² * L² * K³ + C³ * K³)
with max_span constraint L on fan-out 2 nonterminals.
"""

import jax.numpy as jnp

from ._log_semiring import NEG_INF, logsumexp


def mcfg_inside(cg, tw):
    """Inside algorithm for MCFG with fan-out ≤ 2.

    For fan-out 1 nonterminals: alpha1[A, i, j] — same as SCFG.
    For fan-out 2 nonterminals: alpha2[A, i1, j1, i2, j2] — two ranges.
    Constraint: i1 ≤ j1 < i2 ≤ j2.

    When a nonterminal has max_span set, each component's span is bounded
    by that value, reducing complexity for fan-out 2 from O(C⁶) to O(C²L²).

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

    # Extract max_span per nonterminal (-1 means unlimited → use C)
    ms = {}
    if cg.max_spans is not None:
        ms = {int(k): int(cg.max_spans[k]) for k in range(K)}
    else:
        ms = {int(k): -1 for k in range(K)}

    def _max_span(nt):
        """Return effective max span for nonterminal (C if unlimited)."""
        s = ms.get(nt, -1)
        return C if s < 0 else s

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

    # Fill by increasing total span length.
    # Fan-out 2 rules must be processed before fan-out 1 rules that
    # read alpha2 via concatenation (A(xy) → B(x,y)).
    for total_len in range(2, C + 1):
        # Fan-out 2 rules (must come first — alpha2 feeds into fan-out 1
        # concatenation rules below)
        for r in range(cg.n_rules):
            lhs = int(cg.rule_lhs[r])
            if fo[lhs] != 2:
                continue
            n_rhs = int(cg.rule_n_rhs[r])
            log_w = float(cg.rule_log_weights[r])
            n_emit = int(cg.rule_n_emissions[r])
            comp = int(cg.rule_composition[r]) if cg.rule_composition is not None else 0

            L = _max_span(lhs)

            # Fan-out 2 nonterminals: iterate over valid range quadruples
            for i1 in range(C):
                j1_max = min(i1 + L, C)
                for j1 in range(i1 + 1, j1_max + 1):
                    span1 = j1 - i1
                    need_span2 = total_len - span1
                    if need_span2 < 1 or need_span2 > L:
                        continue
                    for i2 in range(j1, C):
                        j2 = i2 + need_span2
                        if j2 > C:
                            break

                        if n_rhs == 1 and n_emit > 0 and fo[int(cg.rule_rhs[r, 0])] == 2:
                            # Fan-out 2 unary with paired emission
                            rhs0 = int(cg.rule_rhs[r, 0])
                            model_idx = int(cg.rule_emission_model[r, 0])
                            if comp == 1 and tw.paired is not None:
                                # 'll': PK(ax, by) → PK(x, y) [pair(a,b)]
                                # Emit left of comp1 (i1) paired with left of comp2 (i2)
                                if span1 >= 2 and need_span2 >= 2:
                                    emit_w = tw.paired[model_idx, i1, i2]
                                    inner = alpha2[rhs0, i1 + 1, j1, i2 + 1, j2]
                                    s = log_w + emit_w + inner
                                    alpha2 = alpha2.at[lhs, i1, j1, i2, j2].set(
                                        jnp.logaddexp(
                                            alpha2[lhs, i1, j1, i2, j2], s))
                            elif comp == 2 and tw.paired is not None:
                                # 'rr': PK(xa, yb) → PK(x, y) [pair(a,b)]
                                # Emit right of comp1 (j1-1) paired with right of comp2 (j2-1)
                                if span1 >= 2 and need_span2 >= 2:
                                    emit_w = tw.paired[model_idx, j1 - 1, j2 - 1]
                                    inner = alpha2[rhs0, i1, j1 - 1, i2, j2 - 1]
                                    s = log_w + emit_w + inner
                                    alpha2 = alpha2.at[lhs, i1, j1, i2, j2].set(
                                        jnp.logaddexp(
                                            alpha2[lhs, i1, j1, i2, j2], s))

                        elif n_rhs == 2:
                            rhs0 = int(cg.rule_rhs[r, 0])
                            rhs1 = int(cg.rule_rhs[r, 1])
                            # Crossing composition: A(x₁y₁, x₂y₂) → B(x₁, x₂) C(y₁, y₂)
                            if fo[rhs0] == 2 and fo[rhs1] == 2:
                                L0 = _max_span(rhs0)
                                L1 = _max_span(rhs1)
                                for k1 in range(i1, j1 + 1):
                                    if k1 - i1 > L0 or j1 - k1 > L1:
                                        continue
                                    for k2 in range(i2, j2 + 1):
                                        if k2 - i2 > L0 or j2 - k2 > L1:
                                            continue
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
                        if n_emit > 0:
                            model_idx = int(cg.rule_emission_model[r, 0])
                            n_pos = int(cg.rule_emission_npos[r, 0])
                            if n_pos == 2 and tw.paired is not None and total_len >= 2:
                                # Paired emission: A → e₁ B e₂
                                emit_w = tw.paired[model_idx, i, j - 1]
                                score = log_w + emit_w + alpha1[rhs0, i + 1, j - 1]
                                alpha1 = alpha1.at[lhs, i, j].set(
                                    jnp.logaddexp(alpha1[lhs, i, j], score))
                            elif n_pos == 1:
                                # Left-emitting unary: A → e B
                                emit_w = tw.single[model_idx, i]
                                score = log_w + emit_w + alpha1[rhs0, i + 1, j]
                                alpha1 = alpha1.at[lhs, i, j].set(
                                    jnp.logaddexp(alpha1[lhs, i, j], score))
                        else:
                            # Non-emitting unary: A → B
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

    log_likelihood = alpha1[cg.start, 0, C]
    return alpha1, alpha2, log_likelihood


def mcfg_viterbi(cg, tw):
    """CYK/Viterbi for MCFG with full backtracing.

    Args:
        cg: CompiledGrammar
        tw: TerminalWeights

    Returns:
        labels: (C,) int32 — per-column nonterminal labels
        log_prob: scalar
        bp: dict of backpointers
    """
    K = cg.n_nonterminals
    C = tw.C

    v1 = jnp.full((K, C + 1, C + 1), NEG_INF)
    v2 = jnp.full((K, C + 1, C + 1, C + 1, C + 1), NEG_INF)

    fo = {int(k): int(cg.fan_outs[k]) for k in range(K)}

    ms = {}
    if cg.max_spans is not None:
        ms = {int(k): int(cg.max_spans[k]) for k in range(K)}
    else:
        ms = {int(k): -1 for k in range(K)}

    def _max_span(nt):
        s = ms.get(nt, -1)
        return C if s < 0 else s

    bp = {}  # backpointers

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
                    score = log_w + float(tw.single[model_idx, i])
                    if score > float(v1[lhs, i, i + 1]):
                        v1 = v1.at[lhs, i, i + 1].set(score)
                        bp[('fo1', lhs, i, i + 1)] = ('terminal', model_idx)

    # Fill by increasing total span length.
    # Fan-out 2 rules must be processed before fan-out 1 rules that
    # read v2 via concatenation (A(xy) → B(x,y)).
    for total_len in range(2, C + 1):
        # Fan-out 2 rules (must come first)
        for r in range(cg.n_rules):
            lhs = int(cg.rule_lhs[r])
            if fo[lhs] != 2:
                continue
            n_rhs = int(cg.rule_n_rhs[r])
            log_w = float(cg.rule_log_weights[r])
            n_emit = int(cg.rule_n_emissions[r])
            comp = int(cg.rule_composition[r]) if cg.rule_composition is not None else 0

            L = _max_span(lhs)

            for i1 in range(C):
                j1_max = min(i1 + L, C)
                for j1 in range(i1 + 1, j1_max + 1):
                    span1 = j1 - i1
                    need_span2 = total_len - span1
                    if need_span2 < 1 or need_span2 > L:
                        continue
                    for i2 in range(j1, C):
                        j2 = i2 + need_span2
                        if j2 > C:
                            break

                        if n_rhs == 1 and n_emit > 0 and fo[int(cg.rule_rhs[r, 0])] == 2:
                            rhs0 = int(cg.rule_rhs[r, 0])
                            model_idx = int(cg.rule_emission_model[r, 0])
                            if comp == 1 and tw.paired is not None:
                                # 'll': pair left of comp1 with left of comp2
                                if span1 >= 2 and need_span2 >= 2:
                                    emit_w = float(tw.paired[model_idx, i1, i2])
                                    inner = float(v2[rhs0, i1 + 1, j1, i2 + 1, j2])
                                    s = log_w + emit_w + inner
                                    if s > float(v2[lhs, i1, j1, i2, j2]):
                                        v2 = v2.at[lhs, i1, j1, i2, j2].set(s)
                                        bp[('fo2', lhs, i1, j1, i2, j2)] = (
                                            'll_pair', rhs0, model_idx)
                            elif comp == 2 and tw.paired is not None:
                                # 'rr': pair right of comp1 with right of comp2
                                if span1 >= 2 and need_span2 >= 2:
                                    emit_w = float(tw.paired[model_idx, j1 - 1, j2 - 1])
                                    inner = float(v2[rhs0, i1, j1 - 1, i2, j2 - 1])
                                    s = log_w + emit_w + inner
                                    if s > float(v2[lhs, i1, j1, i2, j2]):
                                        v2 = v2.at[lhs, i1, j1, i2, j2].set(s)
                                        bp[('fo2', lhs, i1, j1, i2, j2)] = (
                                            'rr_pair', rhs0, model_idx)

                        elif n_rhs == 2:
                            rhs0 = int(cg.rule_rhs[r, 0])
                            rhs1 = int(cg.rule_rhs[r, 1])
                            if fo[rhs0] == 2 and fo[rhs1] == 2:
                                L0 = _max_span(rhs0)
                                L1 = _max_span(rhs1)
                                for k1 in range(i1, j1 + 1):
                                    if k1 - i1 > L0 or j1 - k1 > L1:
                                        continue
                                    for k2 in range(i2, j2 + 1):
                                        if k2 - i2 > L0 or j2 - k2 > L1:
                                            continue
                                        s = (log_w +
                                             float(v2[rhs0, i1, k1, i2, k2]) +
                                             float(v2[rhs1, k1, j1, k2, j2]))
                                        if s > float(v2[lhs, i1, j1, i2, j2]):
                                            v2 = v2.at[lhs, i1, j1, i2, j2].set(s)
                                            bp[('fo2', lhs, i1, j1, i2, j2)] = (
                                                'cross', rhs0, rhs1, k1, k2)

                            elif fo[rhs0] == 1 and fo[rhs1] == 1:
                                s = (log_w +
                                     float(v1[rhs0, i1, j1]) +
                                     float(v1[rhs1, i2, j2]))
                                if s > float(v2[lhs, i1, j1, i2, j2]):
                                    v2 = v2.at[lhs, i1, j1, i2, j2].set(s)
                                    bp[('fo2', lhs, i1, j1, i2, j2)] = (
                                        'pair', rhs0, rhs1)

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
                            score = log_w + float(v1[rhs0, i, k]) + float(v1[rhs1, k, j])
                            if score > float(v1[lhs, i, j]):
                                v1 = v1.at[lhs, i, j].set(score)
                                bp[('fo1', lhs, i, j)] = ('binary', rhs0, rhs1, k)

                elif n_rhs == 1:
                    rhs0 = int(cg.rule_rhs[r, 0])
                    if fo[rhs0] == 1:
                        if n_emit > 0:
                            model_idx = int(cg.rule_emission_model[r, 0])
                            n_pos = int(cg.rule_emission_npos[r, 0])
                            if n_pos == 2 and tw.paired is not None and total_len >= 2:
                                emit_w = float(tw.paired[model_idx, i, j - 1])
                                inner = float(v1[rhs0, i + 1, j - 1])
                                score = log_w + emit_w + inner
                                if score > float(v1[lhs, i, j]):
                                    v1 = v1.at[lhs, i, j].set(score)
                                    bp[('fo1', lhs, i, j)] = ('paired', rhs0, i + 1, j - 1, model_idx)
                            elif n_pos == 1:
                                # Left-emitting unary: A → e B
                                emit_w = float(tw.single[model_idx, i])
                                inner = float(v1[rhs0, i + 1, j])
                                score = log_w + emit_w + inner
                                if score > float(v1[lhs, i, j]):
                                    v1 = v1.at[lhs, i, j].set(score)
                                    bp[('fo1', lhs, i, j)] = ('emit_unary', rhs0, i + 1, j, model_idx)
                        else:
                            # Non-emitting unary: A → B
                            score = log_w + float(v1[rhs0, i, j])
                            if score > float(v1[lhs, i, j]):
                                v1 = v1.at[lhs, i, j].set(score)
                                bp[('fo1', lhs, i, j)] = ('unary', rhs0)

                    elif fo[rhs0] == 2:
                        # Fan-out 2 → fan-out 1: A(xy) → B(x, y)
                        for i2 in range(i + 1, j):
                            score = log_w + float(v2[rhs0, i, i2, i2, j])
                            if score > float(v1[lhs, i, j]):
                                v1 = v1.at[lhs, i, j].set(score)
                                bp[('fo1', lhs, i, j)] = ('concat', rhs0, i, i2, i2, j)

    log_prob = float(v1[cg.start, 0, C])

    # Traceback
    labels = jnp.zeros(C, dtype=jnp.int32)

    def _trace1(A, i, j):
        nonlocal labels
        if i >= j:
            return
        key = ('fo1', A, i, j)
        if key not in bp:
            return
        info = bp[key]
        if info[0] == 'terminal':
            labels = labels.at[i].set(A)
        elif info[0] == 'emit_unary':
            _, rhs, ci, cj, _ = info
            labels = labels.at[i].set(A)
            _trace1(rhs, ci, cj)
        elif info[0] == 'paired':
            _, rhs, ci, cj, _ = info
            labels = labels.at[i].set(A)
            labels = labels.at[j - 1].set(A)
            _trace1(rhs, ci, cj)
        elif info[0] == 'binary':
            _, rhs0, rhs1, k = info
            _trace1(rhs0, i, k)
            _trace1(rhs1, k, j)
        elif info[0] == 'unary':
            _, rhs = info
            _trace1(rhs, i, j)
        elif info[0] == 'concat':
            _, rhs0, ci1, cj1, ci2, cj2 = info
            _trace2(rhs0, ci1, cj1, ci2, cj2)

    def _trace2(A, i1, j1, i2, j2):
        nonlocal labels
        key = ('fo2', A, i1, j1, i2, j2)
        if key not in bp:
            return
        info = bp[key]
        if info[0] == 'pair':
            _, rhs0, rhs1 = info
            _trace1(rhs0, i1, j1)
            _trace1(rhs1, i2, j2)
        elif info[0] == 'cross':
            _, rhs0, rhs1, k1, k2 = info
            _trace2(rhs0, i1, k1, i2, k2)
            _trace2(rhs1, k1, j1, k2, j2)
        elif info[0] == 'll_pair':
            # Left-left: emitted i1 and i2, inner is [i1+1,j1) [i2+1,j2)
            _, rhs0, _ = info
            labels = labels.at[i1].set(A)
            labels = labels.at[i2].set(A)
            _trace2(rhs0, i1 + 1, j1, i2 + 1, j2)
        elif info[0] == 'rr_pair':
            # Right-right: emitted j1-1 and j2-1, inner is [i1,j1-1) [i2,j2-1)
            _, rhs0, _ = info
            labels = labels.at[j1 - 1].set(A)
            labels = labels.at[j2 - 1].set(A)
            _trace2(rhs0, i1, j1 - 1, i2, j2 - 1)

    _trace1(cg.start, 0, C)

    return labels, log_prob, bp
