"""Tests for xrate .eg grammar file parser."""

import math
import os

import jax.numpy as jnp
import numpy as np
import pytest

from jaxrate.xrate_parser import parse_xrate, parse_xrate_file, parse_sexpr
from jaxrate.grammar import classify_grammar, validate_grammar, compile_grammar
from jaxrate.types import TerminalWeights


DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'jaxrate', 'data')


# ---------------------------------------------------------------------------
# S-expression parser tests
# ---------------------------------------------------------------------------

class TestSexprParser:
    def test_simple_atom(self):
        result = parse_sexpr("hello")
        assert result == ["hello"]

    def test_simple_list(self):
        result = parse_sexpr("(a b c)")
        assert result == [["a", "b", "c"]]

    def test_nested(self):
        result = parse_sexpr("(a (b c) d)")
        assert result == [["a", ["b", "c"], "d"]]

    def test_numbers(self):
        result = parse_sexpr("(prob 0.25)")
        assert result == [["prob", 0.25]]

    def test_integers(self):
        result = parse_sexpr("(count 42)")
        assert result == [["count", 42]]

    def test_comments(self):
        result = parse_sexpr(";; comment\n(a b)")
        assert result == [["a", "b"]]

    def test_multiple_top_level(self):
        result = parse_sexpr("(a) (b)")
        assert result == [["a"], ["b"]]

    def test_empty_list(self):
        result = parse_sexpr("()")
        assert result == [[]]


# ---------------------------------------------------------------------------
# nullrna.eg tests
# ---------------------------------------------------------------------------

class TestNullRNA:
    @pytest.fixture
    def nullrna(self):
        return parse_xrate_file(os.path.join(DATA_DIR, 'nullrna.eg'))

    def test_alphabet(self, nullrna):
        assert nullrna.alphabet['name'] == 'RNA'
        assert nullrna.alphabet['tokens'] == ['a', 'c', 'g', 'u']

    def test_chain(self, nullrna):
        assert len(nullrna.chains) == 1
        chain = nullrna.chains[0]
        assert chain['terminals'] == ['X']
        assert chain['n_positions'] == 1
        assert chain['update_policy'] == 'rev'
        assert chain['alphabet_size'] == 4
        np.testing.assert_allclose(chain['pi'].sum(), 1.0, atol=1e-10)

    def test_grammar_nonterminals(self, nullrna):
        g = nullrna.grammar
        nt_names = [nt.name for nt in g.nonterminals]
        assert 'S' in nt_names
        assert 'S*' in nt_names

    def test_grammar_classification(self, nullrna):
        assert classify_grammar(nullrna.grammar) == 'hmm'

    def test_grammar_validates(self, nullrna):
        validate_grammar(nullrna.grammar)

    def test_grammar_compiles(self, nullrna):
        cg = compile_grammar(nullrna.grammar)
        assert cg.grammar_class == 'hmm'

    def test_rate_matrix_diagonal(self, nullrna):
        Q = nullrna.chains[0]['rate_matrix']
        # Diagonal should be negative of row sums
        for i in range(Q.shape[0]):
            off_diag_sum = Q[i].sum() - Q[i, i]
            np.testing.assert_allclose(Q[i, i], -off_diag_sum, atol=1e-10)

    def test_rate_matrix_values(self, nullrna):
        chain = nullrna.chains[0]
        Q = chain['rate_matrix']
        # a→c rate = 0.099 from the file
        tok_idx = {t: i for i, t in enumerate(nullrna.alphabet['tokens'])}
        assert abs(Q[tok_idx['a'], tok_idx['c']] - 0.099) < 1e-10


# ---------------------------------------------------------------------------
# pfold.eg tests
# ---------------------------------------------------------------------------

class TestPfold:
    @pytest.fixture
    def pfold(self):
        return parse_xrate_file(os.path.join(DATA_DIR, 'pfold.eg'))

    def test_alphabet(self, pfold):
        assert pfold.alphabet['tokens'] == ['a', 'c', 'g', 'u']

    def test_chains(self, pfold):
        assert len(pfold.chains) == 2
        single_chains = [c for c in pfold.chains if c['n_positions'] == 1]
        paired_chains = [c for c in pfold.chains if c['n_positions'] == 2]
        assert len(single_chains) == 1
        assert len(paired_chains) == 1

    def test_paired_chain(self, pfold):
        paired = [c for c in pfold.chains if c['n_positions'] == 2][0]
        assert paired['terminals'] == ['LNUC', 'RNUC']
        assert paired['rate_matrix'].shape == (16, 16)
        assert paired['pi'].shape == (16,)
        np.testing.assert_allclose(paired['pi'].sum(), 1.0, atol=1e-10)

    def test_single_chain(self, pfold):
        single = [c for c in pfold.chains if c['n_positions'] == 1][0]
        assert single['terminals'] == ['NUC']
        assert single['rate_matrix'].shape == (4, 4)
        np.testing.assert_allclose(single['pi'].sum(), 1.0, atol=1e-10)

    def test_nonterminals(self, pfold):
        g = pfold.grammar
        nt_names = [nt.name for nt in g.nonterminals]
        assert 'pfoldS' in nt_names
        assert 'pfoldF' in nt_names
        assert 'pfoldF*' in nt_names
        assert 'pfoldL' in nt_names
        assert 'pfoldB' in nt_names
        assert 'pfoldU' in nt_names
        assert 'pfoldU*' in nt_names

    def test_classification(self, pfold):
        assert classify_grammar(pfold.grammar) == 'scfg'

    def test_validates(self, pfold):
        validate_grammar(pfold.grammar)

    def test_compiles(self, pfold):
        cg = compile_grammar(pfold.grammar)
        assert cg.grammar_class == 'scfg'

    def test_bifurcation_rule(self, pfold):
        g = pfold.grammar
        nt_names = [nt.name for nt in g.nonterminals]
        # pfoldB → pfoldL pfoldS is a bifurcation
        for r in g.rules:
            if g.nonterminals[r.lhs].name == 'pfoldB':
                rhs_names = [g.nonterminals[i].name for i in r.rhs]
                assert rhs_names == ['pfoldL', 'pfoldS']
                assert len(r.emissions) == 0
                break
        else:
            pytest.fail("No pfoldB rule found")

    def test_paired_emission_rule(self, pfold):
        g = pfold.grammar
        # pfoldF → LNUC pfoldF* RNUC
        for r in g.rules:
            if g.nonterminals[r.lhs].name == 'pfoldF':
                assert len(r.emissions) == 1
                assert r.emissions[0].n_positions == 2
                rhs_names = [g.nonterminals[i].name for i in r.rhs]
                assert 'pfoldF*' in rhs_names
                break
        else:
            pytest.fail("No pfoldF paired emission rule found")

    def test_single_emission_rule(self, pfold):
        g = pfold.grammar
        # pfoldU → NUC pfoldU*
        for r in g.rules:
            if g.nonterminals[r.lhs].name == 'pfoldU':
                assert len(r.emissions) == 1
                assert r.emissions[0].n_positions == 1
                rhs_names = [g.nonterminals[i].name for i in r.rhs]
                assert 'pfoldU*' in rhs_names
                break
        else:
            pytest.fail("No pfoldU single emission rule found")

    def test_termination_rule(self, pfold):
        g = pfold.grammar
        # pfoldU* → ()
        for r in g.rules:
            if g.nonterminals[r.lhs].name == 'pfoldU*' and len(r.rhs) == 0:
                assert len(r.emissions) == 0
                break
        else:
            pytest.fail("No pfoldU* termination rule found")

    def test_model_mapping(self, pfold):
        assert ('NUC',) in pfold.chain_to_model
        assert ('LNUC', 'RNUC') in pfold.chain_to_model
        assert pfold.chain_to_model[('NUC',)][0] == 'single'
        assert pfold.chain_to_model[('LNUC', 'RNUC')][0] == 'paired'

    def test_run_inside_with_synthetic_weights(self, pfold):
        """Smoke test: run inside algorithm on parsed pfold grammar."""
        from jaxrate.inside import inside

        g = pfold.grammar
        C = 8
        np.random.seed(123)
        single = jnp.array(np.random.randn(1, C) - 1.0)
        paired = jnp.array(np.random.randn(1, C, C) - 2.0)
        tw = TerminalWeights(single=single, paired=paired, C=C)

        chart, log_ll = inside(g, tw)
        assert jnp.isfinite(log_ll)
        assert float(log_ll) < 0  # should be negative log-likelihood

    def test_run_viterbi_with_synthetic_weights(self, pfold):
        """Smoke test: run Viterbi on parsed pfold grammar."""
        from jaxrate.viterbi import viterbi

        g = pfold.grammar
        C = 6
        np.random.seed(456)
        single = jnp.array(np.random.randn(1, C) - 1.0)
        paired = jnp.array(np.random.randn(1, C, C) - 2.0)
        tw = TerminalWeights(single=single, paired=paired, C=C)

        labels, log_prob = viterbi(g, tw)
        assert labels.shape == (C,)
        assert jnp.isfinite(log_prob)


# ---------------------------------------------------------------------------
# Subby model conversion tests
# ---------------------------------------------------------------------------

class TestSubbyConversion:
    @pytest.fixture
    def pfold(self):
        return parse_xrate_file(os.path.join(DATA_DIR, 'pfold.eg'))

    def test_to_subby_models(self, pfold):
        models = pfold.to_subby_models()
        assert len(models['single']) == 1
        assert len(models['paired']) == 1

    def test_single_model_shape(self, pfold):
        models = pfold.to_subby_models()
        single = models['single'][0]
        # DiagModel has eigenvalues, eigenvectors, pi
        assert single.pi.shape == (4,)
        assert single.eigenvalues.shape == (4,)
        assert single.eigenvectors.shape == (4, 4)

    def test_paired_model_shape(self, pfold):
        models = pfold.to_subby_models()
        paired = models['paired'][0]
        # paired is a dict with 'model' and 'A'
        assert paired['A'] == 4
        model = paired['model']
        assert model.pi.shape == (16,)


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

class TestEdgeCases:
    def test_parse_inline_string(self):
        text = """
        (alphabet (name RNA) (token (a c g u)))
        (grammar
         (name simple)
         (chain
          (terminal (X))
          (initial (state (a)) (prob 0.5))
          (initial (state (c)) (prob 0.5))
          (mutate (from (a)) (to (c)) (rate 1.0))
          (mutate (from (c)) (to (a)) (rate 1.0)))
         (transform (from (S)) (to (X S*)))
         (transform (from (S*)) (to ()) (prob 1)))
        """
        result = parse_xrate(text)
        assert len(result.grammar.nonterminals) == 2
        assert len(result.grammar.rules) == 2

    def test_no_grammar_block_raises(self):
        with pytest.raises(ValueError, match="No.*grammar.*block"):
            parse_xrate("(alphabet (name RNA) (token (a c g u)))")

    def test_custom_start_nonterminal(self):
        text = """
        (grammar
         (name test)
         (chain (terminal (X))
          (initial (state (a)) (prob 1.0)))
         (transform (from (A)) (to (X B)))
         (transform (from (B)) (to ()) (prob 1))
         (transform (from (C)) (to (A)) (prob 1)))
        """
        result = parse_xrate(text, start_nonterminal='C')
        assert result.grammar.nonterminals[result.grammar.start].name == 'C'

    def test_invalid_start_nonterminal_raises(self):
        text = """
        (grammar
         (name test)
         (chain (terminal (X))
          (initial (state (a)) (prob 1.0)))
         (transform (from (S)) (to (X S*)))
         (transform (from (S*)) (to ()) (prob 1)))
        """
        with pytest.raises(ValueError, match="not found"):
            parse_xrate(text, start_nonterminal='NONEXISTENT')
