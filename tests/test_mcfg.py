"""Tests for MCFG algorithms."""

import pytest
import jax.numpy as jnp
import numpy as np

from jaxrate import (
    compile_grammar, TerminalWeights, inside, viterbi,
)
from jaxrate.presets import pseudoknot_grammar
from jaxrate.mcfg import mcfg_inside


class TestMCFGInside:
    def test_basic(self):
        grammar = pseudoknot_grammar()
        cg = compile_grammar(grammar)

        C = 4
        single = jnp.array([[-1.0, -1.5, -2.0, -1.0]])
        paired = jnp.full((1, C, C), -3.0)
        tw = TerminalWeights(single=single, paired=paired, C=C)

        alpha1, alpha2, ll = mcfg_inside(cg, tw)

        K = cg.n_nonterminals
        assert alpha1.shape == (K, C + 1, C + 1)
        assert alpha2.shape == (K, C + 1, C + 1, C + 1, C + 1)
        # Log-likelihood might be -inf for very small grammars, but should be finite
        # or -inf (no valid parse)

    def test_classification(self):
        grammar = pseudoknot_grammar()
        cg = compile_grammar(grammar)
        assert cg.grammar_class == 'mcfg'


class TestMCFGDispatch:
    def test_inside_dispatch(self):
        grammar = pseudoknot_grammar()
        C = 4
        single = jnp.array([[-1.0, -1.5, -2.0, -1.0]])
        paired = jnp.full((1, C, C), -3.0)
        tw = TerminalWeights(single=single, paired=paired, C=C)

        chart, ll = inside(grammar, tw)
        # For MCFG, chart is (alpha1, alpha2)
        assert isinstance(chart, tuple)
        assert len(chart) == 2
