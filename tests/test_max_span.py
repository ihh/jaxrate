"""Tests for max_span constraint on MCFG nonterminals."""

import pytest
import jax.numpy as jnp
import numpy as np

from jaxrate import (
    compile_grammar, validate_grammar, TerminalWeights, viterbi,
    GrammarBuilder, Nonterminal, EmissionGroup,
)
from jaxrate.presets import pseudoknot_grammar
from jaxrate.mcfg import mcfg_inside, mcfg_viterbi


class TestNonterminalMaxSpan:
    def test_default_none(self):
        nt = Nonterminal('S')
        assert nt.max_span is None

    def test_explicit_max_span(self):
        nt = Nonterminal('PK', fan_out=2, max_span=10)
        assert nt.max_span == 10
        assert nt.fan_out == 2

    def test_none_max_span(self):
        nt = Nonterminal('S', max_span=None)
        assert nt.max_span is None


class TestCompiledGrammarMaxSpans:
    def test_max_spans_array(self):
        grammar = pseudoknot_grammar(max_span=10)
        cg = compile_grammar(grammar)
        assert cg.max_spans is not None
        assert cg.max_spans.shape == (cg.n_nonterminals,)
        # S has no max_span → -1, PK has max_span=10, L has no max_span → -1
        assert int(cg.max_spans[0]) == -1   # S
        assert int(cg.max_spans[1]) == 10   # PK
        assert int(cg.max_spans[2]) == -1   # L

    def test_no_max_span(self):
        grammar = pseudoknot_grammar()
        cg = compile_grammar(grammar)
        # All should be -1 (unlimited)
        for k in range(cg.n_nonterminals):
            assert int(cg.max_spans[k]) == -1


class TestValidation:
    def test_rejects_max_span_zero(self):
        gb = GrammarBuilder()
        gb.add_nonterminal('S', max_span=0)
        gb.add_rule(0, rhs=[], emissions=[EmissionGroup(1, 0)])
        grammar = gb.build(n_models=1)
        with pytest.raises(ValueError, match="max_span must be >= 1"):
            validate_grammar(grammar)

    def test_rejects_negative_max_span(self):
        gb = GrammarBuilder()
        gb.add_nonterminal('S', max_span=-5)
        gb.add_rule(0, rhs=[], emissions=[EmissionGroup(1, 0)])
        grammar = gb.build(n_models=1)
        with pytest.raises(ValueError, match="max_span must be >= 1"):
            validate_grammar(grammar)

    def test_accepts_max_span_one(self):
        gb = GrammarBuilder()
        gb.add_nonterminal('S', max_span=1)
        gb.add_rule(0, rhs=[], emissions=[EmissionGroup(1, 0)])
        grammar = gb.build(n_models=1)
        validate_grammar(grammar)  # should not raise


class TestMaxSpanConstraint:
    def _make_tw(self, C):
        """Create terminal weights with moderate pair signals."""
        np.random.seed(42)
        single = jnp.array(np.random.randn(1, C) - 1.0)
        paired = jnp.array(np.random.randn(1, C, C) - 2.0)
        return TerminalWeights(single=single, paired=paired, C=C)

    def test_large_max_span_equals_unconstrained(self):
        """max_span >= C should give the same result as unconstrained."""
        C = 6
        tw = self._make_tw(C)

        grammar_none = pseudoknot_grammar()
        cg_none = compile_grammar(grammar_none)
        _, _, ll_none = mcfg_inside(cg_none, tw)

        grammar_large = pseudoknot_grammar(max_span=C)
        cg_large = compile_grammar(grammar_large)
        _, _, ll_large = mcfg_inside(cg_large, tw)

        assert jnp.isclose(ll_none, ll_large, atol=1e-10), \
            f"max_span={C} should equal unconstrained: {ll_large} vs {ll_none}"

    def test_small_max_span_restricts_ll(self):
        """Small max_span should give LL ≤ unconstrained LL."""
        C = 6
        tw = self._make_tw(C)

        grammar_none = pseudoknot_grammar()
        cg_none = compile_grammar(grammar_none)
        _, _, ll_none = mcfg_inside(cg_none, tw)

        grammar_small = pseudoknot_grammar(max_span=2)
        cg_small = compile_grammar(grammar_small)
        _, _, ll_small = mcfg_inside(cg_small, tw)

        assert ll_small <= ll_none + 1e-10, \
            f"Constrained LL should be ≤ unconstrained: {ll_small} vs {ll_none}"


class TestMCFGViterbi:
    def _make_tw(self, C):
        np.random.seed(42)
        single = jnp.array(np.random.randn(1, C) - 1.0)
        paired = jnp.array(np.random.randn(1, C, C) - 2.0)
        return TerminalWeights(single=single, paired=paired, C=C)

    def test_viterbi_returns_labels(self):
        C = 4
        tw = self._make_tw(C)
        grammar = pseudoknot_grammar()
        cg = compile_grammar(grammar)
        labels, log_prob, bp = mcfg_viterbi(cg, tw)
        assert labels.shape == (C,)
        assert isinstance(bp, dict)

    def test_viterbi_with_max_span(self):
        C = 6
        tw = self._make_tw(C)
        grammar = pseudoknot_grammar(max_span=3)
        cg = compile_grammar(grammar)
        labels, log_prob, bp = mcfg_viterbi(cg, tw)
        assert labels.shape == (C,)
        # All labels should be valid nonterminal indices
        assert jnp.all(labels >= 0)
        assert jnp.all(labels < cg.n_nonterminals)

    def test_viterbi_dispatch(self):
        """Test that the viterbi dispatcher works with MCFG."""
        C = 4
        tw = self._make_tw(C)
        grammar = pseudoknot_grammar()
        labels, log_prob = viterbi(grammar, tw)
        assert labels.shape == (C,)

    def test_viterbi_strong_pair_signal(self):
        """With strong pair signals, Viterbi should find paired structure."""
        # C=4 is a valid length for pseudoknot_grammar (S → L S, L span 3 + S span 1)
        C = 4
        single = jnp.full((2, C), -2.0)
        paired = jnp.full((2, C, C), -10.0)
        # Strong pair signal at (0, 2) for model 1 (paired emission)
        paired = paired.at[1, 0, 2].set(-0.1)
        tw = TerminalWeights(single=single, paired=paired, C=C)

        grammar = pseudoknot_grammar()
        labels, log_prob, bp = mcfg_viterbi(compile_grammar(grammar), tw)
        assert labels.shape == (C,)
        assert log_prob > -1e30
