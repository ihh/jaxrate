"""Cross-algorithm tests: HMM forward == SCFG inside for right-linear grammars."""

import pytest
import jax.numpy as jnp
import numpy as np
import math

from jaxrate import (
    GrammarBuilder, EmissionGroup, compile_grammar,
    TerminalWeights, inside,
)
from jaxrate.hmm import hmm_forward
from jaxrate.scfg import scfg_inside


def _right_linear_grammar():
    """Build a right-linear grammar (classifies as HMM).

    S → e S  (emit then continue)
    S → e    (emit then stop)
    """
    gb = GrammarBuilder()
    S = gb.add_nonterminal('S')

    gb.add_rule(S, rhs=[S], emissions=[EmissionGroup(1, 0)],
                log_weight=math.log(0.8))
    gb.add_rule(S, rhs=[], emissions=[EmissionGroup(1, 0)],
                log_weight=math.log(0.2))

    return gb.build(start=S, n_models=1)


class TestCrossAlgorithm:
    def test_hmm_vs_scfg_log_likelihood(self):
        """Right-linear grammar should give same LL via HMM and SCFG."""
        grammar = _right_linear_grammar()
        cg = compile_grammar(grammar)
        assert cg.grammar_class == 'hmm'

        C = 5
        single = jnp.array([[-1.0, -2.0, -1.5, -0.5, -1.2]])
        tw = TerminalWeights(single=single, C=C)

        # HMM forward
        alpha_hmm, ll_hmm = hmm_forward(cg, tw)

        # SCFG inside (force it even though classified as HMM)
        alpha_scfg, ll_scfg = scfg_inside(cg, tw)

        np.testing.assert_allclose(float(ll_hmm), float(ll_scfg), atol=1e-4)
