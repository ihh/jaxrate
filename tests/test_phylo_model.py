"""Tests for PhyloModel and integrated EM training."""

import pytest
import math
import numpy as np
import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

from subby.jax.types import Tree

from jaxrate import (
    GrammarBuilder, EmissionGroup, Grammar,
    PhyloModel, phylo_train, phylo_em_step,
    compile_grammar, viterbi, inside,
)


def _make_star_tree(n_leaves, branch_length=0.1):
    """Build a star tree: root + n_leaves children."""
    R = 1 + n_leaves
    parent = np.full(R, -1, dtype=np.int32)
    dist = np.zeros(R, dtype=np.float64)
    for i in range(1, R):
        parent[i] = 0
        dist[i] = branch_length
    return Tree(parentIndex=jnp.array(parent), distanceToParent=jnp.array(dist))


def _jc_rate_matrix(A=4, rate=1.0):
    """Jukes-Cantor rate matrix."""
    Q = np.full((A, A), rate / (A - 1), dtype=np.float64)
    np.fill_diagonal(Q, -rate)
    return Q


def _two_state_phylo_hmm():
    """2-state HMM with different rate matrices per state.

    State 0 ('slow'): low substitution rate
    State 1 ('fast'): high substitution rate
    """
    gb = GrammarBuilder()
    S0 = gb.add_nonterminal('slow')
    S1 = gb.add_nonterminal('fast')

    gb.add_rule(S0, rhs=[S0], emissions=[EmissionGroup(1, 0)],
                log_weight=math.log(0.9))
    gb.add_rule(S0, rhs=[S1], emissions=[EmissionGroup(1, 0)],
                log_weight=math.log(0.09))
    gb.add_rule(S0, rhs=[], emissions=[EmissionGroup(1, 0)],
                log_weight=math.log(0.01))

    gb.add_rule(S1, rhs=[S1], emissions=[EmissionGroup(1, 1)],
                log_weight=math.log(0.9))
    gb.add_rule(S1, rhs=[S0], emissions=[EmissionGroup(1, 1)],
                log_weight=math.log(0.09))
    gb.add_rule(S1, rhs=[], emissions=[EmissionGroup(1, 1)],
                log_weight=math.log(0.01))

    grammar = gb.build(start=S0, n_models=2)

    chains = [
        {'rate_matrix': _jc_rate_matrix(4, rate=0.5), 'pi': np.full(4, 0.25)},
        {'rate_matrix': _jc_rate_matrix(4, rate=5.0), 'pi': np.full(4, 0.25)},
    ]

    return grammar, chains


def _make_alignment(n_leaves=4, C=20, seed=42):
    """Make a synthetic alignment with conserved + variable regions."""
    np.random.seed(seed)
    R = 1 + n_leaves  # root + leaves
    gap = 5

    alignment = np.full((R, C), gap, dtype=np.int32)

    # Conserved region (cols 0-9): all leaves same
    base = np.random.randint(0, 4, size=C)
    for i in range(1, R):
        alignment[i] = base.copy()

    # Variable region (cols 10-19): random substitutions
    for i in range(1, R):
        for c in range(C // 2, C):
            if np.random.random() < 0.7:
                alignment[i, c] = np.random.randint(0, 4)

    return alignment


class TestPhyloModel:
    def test_construction(self):
        grammar, chains = _two_state_phylo_hmm()
        tree = _make_star_tree(4)
        alignment = _make_alignment(4, 20)

        pm = PhyloModel(grammar, alignment, tree, chains)
        assert pm.grammar is grammar
        assert len(pm.chains) == 2
        assert pm.alignment.shape == (5, 20)

    def test_terminal_weights(self):
        grammar, chains = _two_state_phylo_hmm()
        tree = _make_star_tree(4)
        alignment = _make_alignment(4, 20)

        pm = PhyloModel(grammar, alignment, tree, chains)
        tw = pm.terminal_weights()

        assert tw.C == 20
        assert tw.single.shape == (2, 20)
        assert jnp.all(jnp.isfinite(tw.single))

        # Cached: same object returned
        tw2 = pm.terminal_weights()
        assert tw is tw2

        # Recompute: new object
        tw3 = pm.terminal_weights(recompute=True)
        assert tw3 is not tw
        np.testing.assert_allclose(tw.single, tw3.single, atol=1e-10)

    def test_terminal_weights_caching_invalidated_on_update(self):
        grammar, chains = _two_state_phylo_hmm()
        tree = _make_star_tree(4)
        alignment = _make_alignment(4, 20)

        pm = PhyloModel(grammar, alignment, tree, chains)
        tw1 = pm.terminal_weights()

        # Update chains → cache invalidated
        new_chains = [
            {'rate_matrix': _jc_rate_matrix(4, rate=2.0), 'pi': np.full(4, 0.25)},
            {'rate_matrix': _jc_rate_matrix(4, rate=10.0), 'pi': np.full(4, 0.25)},
        ]
        pm.update_chains(new_chains)
        tw2 = pm.terminal_weights()

        # Different rates → different terminal weights
        assert not np.allclose(tw1.single, tw2.single, atol=1e-3)

    def test_viterbi_through_phylo_model(self):
        grammar, chains = _two_state_phylo_hmm()
        tree = _make_star_tree(4)
        alignment = _make_alignment(4, 20)

        pm = PhyloModel(grammar, alignment, tree, chains)
        tw = pm.terminal_weights()

        labels, log_prob = viterbi(grammar, tw)
        assert labels.shape == (20,)
        assert jnp.isfinite(log_prob)

    def test_inside_through_phylo_model(self):
        grammar, chains = _two_state_phylo_hmm()
        tree = _make_star_tree(4)
        alignment = _make_alignment(4, 20)

        pm = PhyloModel(grammar, alignment, tree, chains)
        tw = pm.terminal_weights()

        chart, ll = inside(grammar, tw)
        assert jnp.isfinite(ll)
        assert float(ll) < 0


class TestPhyloEmStep:
    def test_em_step_runs(self):
        grammar, chains = _two_state_phylo_hmm()
        tree = _make_star_tree(4)
        alignment = _make_alignment(4, 20)

        pm = PhyloModel(grammar, alignment, tree, chains)
        ll = phylo_em_step(pm)

        assert np.isfinite(ll)
        assert ll < 0

    def test_em_step_updates_grammar(self):
        grammar, chains = _two_state_phylo_hmm()
        tree = _make_star_tree(4)
        alignment = _make_alignment(4, 20)

        pm = PhyloModel(grammar, alignment, tree, chains)
        old_weights = [r.log_weight for r in pm.grammar.rules]

        phylo_em_step(pm, fit_rules=True, fit_rates=False, fit_pi=False)

        new_weights = [r.log_weight for r in pm.grammar.rules]
        # Weights should change
        assert any(abs(o - n) > 1e-10 for o, n in zip(old_weights, new_weights))

    def test_em_step_updates_rates(self):
        grammar, chains = _two_state_phylo_hmm()
        tree = _make_star_tree(4)
        alignment = _make_alignment(4, 20)

        pm = PhyloModel(grammar, alignment, tree, chains)
        old_Q0 = pm.chains[0]['rate_matrix'].copy()

        phylo_em_step(pm, fit_rules=False, fit_rates=True, fit_pi=False)

        new_Q0 = pm.chains[0]['rate_matrix']
        # Rate matrix should change
        assert not np.allclose(old_Q0, new_Q0, atol=1e-10)
        # Row sums should still be ~0
        np.testing.assert_allclose(new_Q0.sum(axis=1), 0.0, atol=1e-10)

    def test_em_step_updates_pi(self):
        grammar, chains = _two_state_phylo_hmm()
        tree = _make_star_tree(4)
        alignment = _make_alignment(4, 20)

        pm = PhyloModel(grammar, alignment, tree, chains)
        old_pi = pm.chains[0]['pi'].copy()

        # Pi update requires rates update too (pi = stationary dist of new Q)
        phylo_em_step(pm, fit_rules=False, fit_rates=True, fit_pi=True)

        new_pi = pm.chains[0]['pi']
        # pi should change (stationary dist of updated Q differs from uniform)
        assert not np.allclose(old_pi, new_pi, atol=1e-10)
        # pi should still sum to 1
        np.testing.assert_allclose(new_pi.sum(), 1.0, atol=1e-10)

    def test_em_step_fit_nothing(self):
        """fit_rules=False, fit_rates=False, fit_pi=False → no changes."""
        grammar, chains = _two_state_phylo_hmm()
        tree = _make_star_tree(4)
        alignment = _make_alignment(4, 20)

        pm = PhyloModel(grammar, alignment, tree, chains)
        old_weights = [r.log_weight for r in pm.grammar.rules]
        old_Q0 = pm.chains[0]['rate_matrix'].copy()
        old_pi = pm.chains[0]['pi'].copy()

        phylo_em_step(pm, fit_rules=False, fit_rates=False, fit_pi=False)

        new_weights = [r.log_weight for r in pm.grammar.rules]
        assert old_weights == new_weights
        np.testing.assert_array_equal(old_Q0, pm.chains[0]['rate_matrix'])
        np.testing.assert_array_equal(old_pi, pm.chains[0]['pi'])


class TestPhyloTrain:
    def test_train_converges(self):
        grammar, chains = _two_state_phylo_hmm()
        tree = _make_star_tree(4)
        alignment = _make_alignment(4, 20)

        pm = PhyloModel(grammar, alignment, tree, chains)
        history = phylo_train(pm, n_iterations=10)

        assert len(history) > 0
        # Log-likelihood should generally increase (or stay same)
        lls = [ll for _, ll in history]
        assert all(np.isfinite(ll) for ll in lls)

    def test_train_rules_only(self):
        grammar, chains = _two_state_phylo_hmm()
        tree = _make_star_tree(4)
        alignment = _make_alignment(4, 20)

        pm = PhyloModel(grammar, alignment, tree, chains)
        old_Q = pm.chains[0]['rate_matrix'].copy()

        history = phylo_train(pm, n_iterations=5,
                              fit_rules=True, fit_rates=False, fit_pi=False)

        assert len(history) > 0
        # Rate matrices unchanged
        np.testing.assert_array_equal(old_Q, pm.chains[0]['rate_matrix'])

    def test_train_rates_only(self):
        grammar, chains = _two_state_phylo_hmm()
        tree = _make_star_tree(4)
        alignment = _make_alignment(4, 20)

        pm = PhyloModel(grammar, alignment, tree, chains)
        old_weights = [r.log_weight for r in pm.grammar.rules]

        history = phylo_train(pm, n_iterations=5,
                              fit_rules=False, fit_rates=True, fit_pi=True)

        assert len(history) > 0
        # Grammar weights unchanged
        new_weights = [r.log_weight for r in pm.grammar.rules]
        assert old_weights == new_weights

    def test_train_all(self):
        grammar, chains = _two_state_phylo_hmm()
        tree = _make_star_tree(4)
        alignment = _make_alignment(4, 20)

        pm = PhyloModel(grammar, alignment, tree, chains)
        old_Q = pm.chains[0]['rate_matrix'].copy()
        old_weights = [r.log_weight for r in pm.grammar.rules]

        history = phylo_train(pm, n_iterations=5,
                              fit_rules=True, fit_rates=True, fit_pi=True)

        assert len(history) > 0
        # Both should have changed
        new_Q = pm.chains[0]['rate_matrix']
        new_weights = [r.log_weight for r in pm.grammar.rules]
        assert not np.allclose(old_Q, new_Q, atol=1e-10)
        assert any(abs(o - n) > 1e-10 for o, n in zip(old_weights, new_weights))

    def test_rate_matrix_stays_valid(self):
        """Rate matrices should remain valid CTMCs after training."""
        grammar, chains = _two_state_phylo_hmm()
        tree = _make_star_tree(4)
        alignment = _make_alignment(4, 20)

        pm = PhyloModel(grammar, alignment, tree, chains)
        phylo_train(pm, n_iterations=5)

        for chain in pm.chains:
            Q = chain['rate_matrix']
            # Row sums = 0
            np.testing.assert_allclose(Q.sum(axis=1), 0.0, atol=1e-10)
            # Off-diagonal >= 0
            mask = ~np.eye(Q.shape[0], dtype=bool)
            assert np.all(Q[mask] >= -1e-15)
            # pi sums to 1 and is positive
            pi = chain['pi']
            np.testing.assert_allclose(pi.sum(), 1.0, atol=1e-10)
            assert np.all(pi > 0)

    def test_rules_only_em_monotonic(self):
        """Rules-only EM should give monotonically increasing LL."""
        grammar, chains = _two_state_phylo_hmm()
        tree = _make_star_tree(4)
        alignment = _make_alignment(4, 20)

        pm = PhyloModel(grammar, alignment, tree, chains)
        history = phylo_train(pm, n_iterations=10,
                              fit_rules=True, fit_rates=False, fit_pi=False)

        lls = [ll for _, ll in history]
        for i in range(len(lls) - 1):
            assert lls[i + 1] >= lls[i] - 1e-8, \
                f"LL decreased at step {i}: {lls[i]:.6f} -> {lls[i+1]:.6f}"
