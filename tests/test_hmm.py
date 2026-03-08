"""Tests for HMM algorithms."""

import pytest
import jax
import jax.numpy as jnp
import numpy as np
import math

from jaxrate import (
    GrammarBuilder, EmissionGroup, compile_grammar,
    TerminalWeights, inside, outside, viterbi,
)
from jaxrate.hmm import hmm_forward, hmm_backward, hmm_viterbi, hmm_posteriors


def _two_state_hmm():
    """Build a simple 2-state HMM.

    State 0: 'low' — prefers model 0
    State 1: 'high' — prefers model 1

    Transitions: self-loop 0.9, switch 0.1
    """
    gb = GrammarBuilder()
    S0 = gb.add_nonterminal('low')
    S1 = gb.add_nonterminal('high')

    # State 0 transitions
    gb.add_rule(S0, rhs=[S0], emissions=[EmissionGroup(1, 0)],
                log_weight=math.log(0.9))
    gb.add_rule(S0, rhs=[S1], emissions=[EmissionGroup(1, 0)],
                log_weight=math.log(0.09))
    gb.add_rule(S0, rhs=[], emissions=[EmissionGroup(1, 0)],
                log_weight=math.log(0.01))

    # State 1 transitions
    gb.add_rule(S1, rhs=[S1], emissions=[EmissionGroup(1, 1)],
                log_weight=math.log(0.9))
    gb.add_rule(S1, rhs=[S0], emissions=[EmissionGroup(1, 1)],
                log_weight=math.log(0.09))
    gb.add_rule(S1, rhs=[], emissions=[EmissionGroup(1, 1)],
                log_weight=math.log(0.01))

    return gb.build(start=S0, n_models=2)


class TestHMMForward:
    def test_basic_log_likelihood(self, simple_terminal_weights):
        grammar = _two_state_hmm()
        cg = compile_grammar(grammar)
        alpha, ll = hmm_forward(cg, simple_terminal_weights)

        assert alpha.shape == (5, 2)
        assert jnp.isfinite(ll)
        assert ll < 0  # log-likelihood should be negative

    def test_single_column(self):
        grammar = _two_state_hmm()
        cg = compile_grammar(grammar)
        tw = TerminalWeights(single=jnp.array([[-1.0], [-2.0]]), C=1)
        alpha, ll = hmm_forward(cg, tw)

        assert alpha.shape == (1, 2)
        assert jnp.isfinite(ll)

    def test_forward_backward_agree(self, simple_terminal_weights):
        grammar = _two_state_hmm()
        cg = compile_grammar(grammar)
        _, ll_fwd = hmm_forward(cg, simple_terminal_weights)
        _, ll_bwd = hmm_backward(cg, simple_terminal_weights)

        np.testing.assert_allclose(float(ll_fwd), float(ll_bwd), atol=1e-6)


class TestHMMViterbi:
    def test_basic_decode(self, simple_terminal_weights):
        grammar = _two_state_hmm()
        cg = compile_grammar(grammar)
        labels, log_prob = hmm_viterbi(cg, simple_terminal_weights)

        assert labels.shape == (5,)
        assert jnp.all((labels >= 0) & (labels < 2))
        assert jnp.isfinite(log_prob)

    def test_viterbi_leq_forward(self, simple_terminal_weights):
        """Viterbi log-prob should be <= forward log-likelihood."""
        grammar = _two_state_hmm()
        cg = compile_grammar(grammar)
        _, ll = hmm_forward(cg, simple_terminal_weights)
        _, vit_prob = hmm_viterbi(cg, simple_terminal_weights)

        assert float(vit_prob) <= float(ll) + 1e-6

    def test_strong_signal_recovery(self):
        """When emission weights strongly favor one state, Viterbi should recover it."""
        grammar = _two_state_hmm()
        cg = compile_grammar(grammar)
        C = 10
        # Columns 0-4: strongly favor state 0; columns 5-9: strongly favor state 1
        single = jnp.zeros((2, C))
        single = single.at[0, :5].set(0.0)   # state 0 good for first half
        single = single.at[1, :5].set(-10.0)  # state 1 bad for first half
        single = single.at[0, 5:].set(-10.0)  # state 0 bad for second half
        single = single.at[1, 5:].set(0.0)   # state 1 good for second half
        tw = TerminalWeights(single=single, C=C)

        labels, _ = hmm_viterbi(cg, tw)
        # First half should be state 0, second half state 1
        assert jnp.all(labels[:5] == 0)
        assert jnp.all(labels[5:] == 1)


class TestHMMPosteriors:
    def test_posteriors_sum_to_one(self, simple_terminal_weights):
        grammar = _two_state_hmm()
        cg = compile_grammar(grammar)
        posteriors, ll = hmm_posteriors(cg, simple_terminal_weights)

        assert posteriors.shape == (5, 2)
        np.testing.assert_allclose(
            posteriors.sum(axis=1), jnp.ones(5), atol=1e-6)

    def test_posteriors_nonnegative(self, simple_terminal_weights):
        grammar = _two_state_hmm()
        cg = compile_grammar(grammar)
        posteriors, _ = hmm_posteriors(cg, simple_terminal_weights)

        assert jnp.all(posteriors >= -1e-10)


class TestHMMDispatch:
    def test_inside_dispatches_to_hmm(self, simple_terminal_weights):
        grammar = _two_state_hmm()
        chart, ll = inside(grammar, simple_terminal_weights)
        # For HMM, chart is alpha (C, K)
        assert chart.shape == (5, 2)
        assert jnp.isfinite(ll)

    def test_viterbi_dispatches_to_hmm(self, simple_terminal_weights):
        grammar = _two_state_hmm()
        labels, log_prob = viterbi(grammar, simple_terminal_weights)
        assert labels.shape == (5,)
