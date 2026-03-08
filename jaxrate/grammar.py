"""Grammar construction, validation, and compilation to JAX arrays."""

import jax.numpy as jnp
import numpy as np

from .types import (
    Grammar, CompiledGrammar, Nonterminal, Rule, EmissionGroup,
)


def classify_grammar(grammar):
    """Classify a grammar as 'hmm', 'scfg', or 'mcfg'.

    - 'hmm': all rules are right-linear (at most one RHS nonterminal,
      appearing at the right end of the composition, fan-out 1)
    - 'scfg': all nonterminals have fan-out 1 but not right-linear
    - 'mcfg': any nonterminal has fan-out > 1
    """
    max_fan_out = max(nt.fan_out for nt in grammar.nonterminals)
    if max_fan_out > 1:
        return 'mcfg'

    # Check if right-linear: every rule has 0 or 1 RHS nonterminals
    for rule in grammar.rules:
        if len(rule.rhs) > 1:
            return 'scfg'

    return 'hmm'


def validate_grammar(grammar):
    """Validate grammar consistency. Raises ValueError on problems."""
    n_nt = len(grammar.nonterminals)
    if n_nt == 0:
        raise ValueError("Grammar must have at least one nonterminal")

    if grammar.start < 0 or grammar.start >= n_nt:
        raise ValueError(
            f"Start nonterminal index {grammar.start} out of range [0, {n_nt})")

    if grammar.nonterminals[grammar.start].fan_out != 1:
        raise ValueError("Start nonterminal must have fan-out 1")

    for i, nt in enumerate(grammar.nonterminals):
        if nt.max_span is not None and nt.max_span < 1:
            raise ValueError(
                f"Nonterminal {i} ({nt.name}): max_span must be >= 1, got {nt.max_span}")

    for i, rule in enumerate(grammar.rules):
        if rule.lhs < 0 or rule.lhs >= n_nt:
            raise ValueError(f"Rule {i}: LHS index {rule.lhs} out of range")
        for j, rhs_idx in enumerate(rule.rhs):
            if rhs_idx < 0 or rhs_idx >= n_nt:
                raise ValueError(
                    f"Rule {i}: RHS index {rhs_idx} at position {j} out of range")
        for j, eg in enumerate(rule.emissions):
            if eg.model_index < 0 or eg.model_index >= grammar.n_models:
                raise ValueError(
                    f"Rule {i}: emission group {j} model_index {eg.model_index} "
                    f"out of range [0, {grammar.n_models})")
            if eg.n_positions < 1:
                raise ValueError(
                    f"Rule {i}: emission group {j} n_positions must be >= 1")


def compile_grammar(grammar):
    """Compile a Grammar into a CompiledGrammar with dense JAX arrays.

    Args:
        grammar: Grammar NamedTuple

    Returns:
        CompiledGrammar with JIT-friendly dense arrays
    """
    validate_grammar(grammar)
    grammar_class = classify_grammar(grammar)

    n_nt = len(grammar.nonterminals)
    n_rules = len(grammar.rules)

    # Compute max dimensions
    max_rhs = max((len(r.rhs) for r in grammar.rules), default=0)
    max_rhs = max(max_rhs, 1)  # at least 1 to avoid zero-dim arrays
    max_emit = max((len(r.emissions) for r in grammar.rules), default=0)
    max_emit = max(max_emit, 1)

    # Build rule arrays
    rule_lhs = np.zeros(n_rules, dtype=np.int32)
    rule_rhs = np.full((n_rules, max_rhs), -1, dtype=np.int32)
    rule_n_rhs = np.zeros(n_rules, dtype=np.int32)
    rule_log_weights = np.zeros(n_rules, dtype=np.float64)
    rule_emission_model = np.full((n_rules, max_emit), -1, dtype=np.int32)
    rule_emission_npos = np.zeros((n_rules, max_emit), dtype=np.int32)
    rule_n_emissions = np.zeros(n_rules, dtype=np.int32)

    for i, rule in enumerate(grammar.rules):
        rule_lhs[i] = rule.lhs
        for j, rhs_idx in enumerate(rule.rhs):
            rule_rhs[i, j] = rhs_idx
        rule_n_rhs[i] = len(rule.rhs)
        rule_log_weights[i] = rule.log_weight
        for j, eg in enumerate(rule.emissions):
            rule_emission_model[i, j] = eg.model_index
            rule_emission_npos[i, j] = eg.n_positions
        rule_n_emissions[i] = len(rule.emissions)

    # Build per-nonterminal rule index lists
    rules_per_nt = [[] for _ in range(n_nt)]
    for i, rule in enumerate(grammar.rules):
        rules_per_nt[rule.lhs].append(i)

    max_rules_per_nt = max((len(lst) for lst in rules_per_nt), default=0)
    max_rules_per_nt = max(max_rules_per_nt, 1)

    rules_for_nt = np.full((n_nt, max_rules_per_nt), -1, dtype=np.int32)
    n_rules_for_nt = np.zeros(n_nt, dtype=np.int32)
    for nt_idx, rule_list in enumerate(rules_per_nt):
        for j, rule_idx in enumerate(rule_list):
            rules_for_nt[nt_idx, j] = rule_idx
        n_rules_for_nt[nt_idx] = len(rule_list)

    fan_outs = np.array([nt.fan_out for nt in grammar.nonterminals], dtype=np.int32)
    max_spans = np.array(
        [nt.max_span if nt.max_span is not None else -1
         for nt in grammar.nonterminals],
        dtype=np.int32)

    return CompiledGrammar(
        grammar_class=grammar_class,
        n_nonterminals=n_nt,
        n_rules=n_rules,
        rule_lhs=jnp.array(rule_lhs),
        rule_rhs=jnp.array(rule_rhs),
        rule_n_rhs=jnp.array(rule_n_rhs),
        rule_log_weights=jnp.array(rule_log_weights),
        rule_emission_model=jnp.array(rule_emission_model),
        rule_emission_npos=jnp.array(rule_emission_npos),
        rule_n_emissions=jnp.array(rule_n_emissions),
        rules_for_nt=jnp.array(rules_for_nt),
        n_rules_for_nt=jnp.array(n_rules_for_nt),
        start=grammar.start,
        n_models=grammar.n_models,
        fan_outs=jnp.array(fan_outs),
        max_spans=jnp.array(max_spans),
    )


class GrammarBuilder:
    """Programmatic grammar construction helper.

    Example:
        gb = GrammarBuilder()
        S = gb.add_nonterminal('S')
        L = gb.add_nonterminal('L')
        gb.add_rule(S, [L, S], [], log_weight=jnp.log(0.6))
        gb.add_rule(S, [], [EmissionGroup(1, 0)], log_weight=jnp.log(0.4))
        gb.add_rule(L, [], [EmissionGroup(2, 1)], log_weight=0.0)
        grammar = gb.build(start=S, n_models=2)
    """

    def __init__(self):
        self.nonterminals = []
        self.rules = []

    def add_nonterminal(self, name, fan_out=1, max_span=None):
        """Add a nonterminal and return its index."""
        idx = len(self.nonterminals)
        self.nonterminals.append(Nonterminal(name=name, fan_out=fan_out, max_span=max_span))
        return idx

    def add_rule(self, lhs, rhs=(), emissions=(), log_weight=0.0, composition=()):
        """Add a production rule.

        Args:
            lhs: LHS nonterminal index
            rhs: tuple/list of RHS nonterminal indices
            emissions: tuple/list of EmissionGroup
            log_weight: log-probability of this rule
            composition: composition function (optional)

        Returns:
            Rule index
        """
        idx = len(self.rules)
        self.rules.append(Rule(
            lhs=lhs,
            rhs=tuple(rhs),
            emissions=tuple(emissions),
            log_weight=float(log_weight),
            composition=tuple(composition),
        ))
        return idx

    def build(self, start=0, n_models=1):
        """Build the Grammar.

        Args:
            start: start nonterminal index
            n_models: number of emission models

        Returns:
            Grammar NamedTuple
        """
        return Grammar(
            nonterminals=list(self.nonterminals),
            rules=list(self.rules),
            start=start,
            n_models=n_models,
        )
