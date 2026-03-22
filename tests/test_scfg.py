"""Tests for SCFG algorithms."""

import pytest
import jax.numpy as jnp
import numpy as np
import math

from jaxrate import (
    GrammarBuilder, EmissionGroup, compile_grammar,
    TerminalWeights, inside, viterbi,
)
from jaxrate.scfg import scfg_inside, scfg_outside, scfg_viterbi, scfg_posteriors


def _simple_scfg():
    """Build a simple SCFG with paired and unpaired emissions.

    S → L S   (stem then more)
    S → e     (unpaired terminal)
    L → e₁ S e₂  (paired emission)
    """
    gb = GrammarBuilder()

    S = gb.add_nonterminal('S')
    L = gb.add_nonterminal('L')

    # S → L S
    gb.add_rule(S, rhs=[L, S], log_weight=math.log(0.3))
    # S → e (unpaired)
    gb.add_rule(S, rhs=[], emissions=[EmissionGroup(1, 0)],
                log_weight=math.log(0.7))
    # L → e₁ S e₂ (paired)
    gb.add_rule(L, rhs=[S],
                emissions=[EmissionGroup(2, 1)],
                log_weight=0.0)

    return gb.build(start=S, n_models=2)


class TestSCFGInside:
    def test_basic(self, paired_terminal_weights):
        grammar = _simple_scfg()
        cg = compile_grammar(grammar)
        alpha, ll = scfg_inside(cg, paired_terminal_weights)

        C = paired_terminal_weights.C
        K = cg.n_nonterminals
        assert alpha.shape == (K, C + 1, C + 1)
        assert jnp.isfinite(ll)
        assert ll < 0

    def test_dispatch(self, paired_terminal_weights):
        grammar = _simple_scfg()
        chart, ll = inside(grammar, paired_terminal_weights)
        assert jnp.isfinite(ll)

    def test_span_1_base_case(self):
        """Span-1 entries should come from terminal rules only."""
        grammar = _simple_scfg()
        cg = compile_grammar(grammar)
        C = 3
        single = jnp.array([[-1.0, -2.0, -1.5], [-3.0, -2.5, -3.0]])
        paired = jnp.full((2, C, C), -5.0)
        tw = TerminalWeights(single=single, paired=paired, C=C)

        alpha, ll = scfg_inside(cg, tw)

        # S is NT 0, rule S→e has log_weight=log(0.7), model 0
        # alpha[S, 0, 1] should be log(0.7) + single[0, 0]
        expected = math.log(0.7) + float(single[0, 0])
        np.testing.assert_allclose(float(alpha[0, 0, 1]), expected, atol=1e-6)


class TestSCFGOutside:
    def test_basic(self, paired_terminal_weights):
        grammar = _simple_scfg()
        cg = compile_grammar(grammar)
        alpha, ll = scfg_inside(cg, paired_terminal_weights)
        beta = scfg_outside(cg, paired_terminal_weights, alpha)

        C = paired_terminal_weights.C
        K = cg.n_nonterminals
        assert beta.shape == (K, C + 1, C + 1)
        # Start symbol full span outside should be 0
        np.testing.assert_allclose(float(beta[cg.start, 0, C]), 0.0, atol=1e-6)


class TestSCFGViterbi:
    def test_basic(self, paired_terminal_weights):
        grammar = _simple_scfg()
        cg = compile_grammar(grammar)
        labels, log_prob, bp = scfg_viterbi(cg, paired_terminal_weights)

        C = paired_terminal_weights.C
        assert labels.shape == (C,)
        assert jnp.isfinite(log_prob)

    def test_dispatch(self, paired_terminal_weights):
        grammar = _simple_scfg()
        labels, log_prob = viterbi(grammar, paired_terminal_weights)
        assert labels.shape == (paired_terminal_weights.C,)


class TestSCFGPosteriors:
    def test_posteriors_shape(self, paired_terminal_weights):
        grammar = _simple_scfg()
        cg = compile_grammar(grammar)
        posteriors, log_ll = scfg_posteriors(cg, paired_terminal_weights)

        C = paired_terminal_weights.C
        K = cg.n_nonterminals
        assert posteriors.shape == (C, K)
        assert jnp.isfinite(log_ll)

    def test_posteriors_finite(self, paired_terminal_weights):
        grammar = _simple_scfg()
        cg = compile_grammar(grammar)
        posteriors, _ = scfg_posteriors(cg, paired_terminal_weights)

        # All posteriors should be finite and non-NaN
        assert jnp.all(jnp.isfinite(posteriors))
        # At least some columns should have non-zero posteriors
        assert posteriors.sum() > 0

    def test_posteriors_nonnegative(self, paired_terminal_weights):
        grammar = _simple_scfg()
        cg = compile_grammar(grammar)
        posteriors, _ = scfg_posteriors(cg, paired_terminal_weights)

        assert jnp.all(posteriors >= -1e-10)
