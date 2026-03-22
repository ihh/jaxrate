"""Tests for xrate .eg grammar file parser."""

import math
import os

import jax.numpy as jnp
import numpy as np
import pytest

from jaxrate.xrate_parser import (
    parse_xrate, parse_xrate_file, parse_sexpr, expand_macros,
    _eval_parametric,
)
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


# ---------------------------------------------------------------------------
# Macro expansion unit tests
# ---------------------------------------------------------------------------

class TestMacroExpansion:
    def test_define(self):
        exprs = parse_sexpr("(&define N 5) (value N)")
        result = expand_macros(exprs)
        assert result == [['value', 5]]

    def test_define_string(self):
        exprs = parse_sexpr("(&define NAME hello) (tag NAME)")
        result = expand_macros(exprs)
        assert result == [['tag', 'hello']]

    def test_foreach_integer(self):
        exprs = parse_sexpr("(&foreach-integer I (1 3) (item I))")
        result = expand_macros(exprs)
        assert result == [['item', 1], ['item', 2], ['item', 3]]

    def test_foreach_integer_with_define(self):
        exprs = parse_sexpr("(&define N 3) (&foreach-integer I (1 N) (item I))")
        result = expand_macros(exprs)
        assert result == [['item', 1], ['item', 2], ['item', 3]]

    def test_foreach_token(self):
        exprs = parse_sexpr(
            "(alphabet (name DNA) (token (a c g t)))"
            " (grammar (&foreach-token X (state X)))")
        result = expand_macros(exprs)
        grammar = result[1]
        states = [x for x in grammar if isinstance(x, list) and x[0] == 'state']
        assert len(states) == 4
        assert states[0] == ['state', 'a']
        assert states[3] == ['state', 't']

    def test_tokens_count(self):
        exprs = parse_sexpr(
            "(alphabet (name RNA) (token (a c g u)))"
            " (count &TOKENS)")
        result = expand_macros(exprs)
        assert result[1] == ['count', 4]

    def test_cat(self):
        exprs = parse_sexpr("((&. hello world))")
        result = expand_macros(exprs)
        assert result == [['helloworld']]

    def test_cat_with_numbers(self):
        exprs = parse_sexpr("((&. S 1))")
        result = expand_macros(exprs)
        assert result == [['S1']]

    def test_sum(self):
        exprs = parse_sexpr("((&+ 1 2 3))")
        result = expand_macros(exprs)
        assert result == [[6]]

    def test_sub(self):
        exprs = parse_sexpr("((&- 10 3))")
        result = expand_macros(exprs)
        assert result == [[7]]

    def test_mul(self):
        exprs = parse_sexpr("((&* 3 4))")
        result = expand_macros(exprs)
        assert result == [[12]]

    def test_div(self):
        exprs = parse_sexpr("((&/ 1 3))")
        result = expand_macros(exprs)
        assert len(result) == 1
        assert len(result[0]) == 1
        assert abs(result[0][0] - 1/3) < 1e-10

    def test_mod(self):
        exprs = parse_sexpr("((&% 7 3))")
        result = expand_macros(exprs)
        assert result == [[1]]

    def test_eq_true(self):
        exprs = parse_sexpr("((&= 5 5))")
        result = expand_macros(exprs)
        assert result == [[1]]

    def test_eq_false(self):
        exprs = parse_sexpr("((&= 5 3))")
        result = expand_macros(exprs)
        assert result == [[0]]

    def test_if_true(self):
        exprs = parse_sexpr("(&? 1 yes no)")
        result = expand_macros(exprs)
        assert result == ['yes']

    def test_if_false(self):
        exprs = parse_sexpr("(&? 0 yes no)")
        result = expand_macros(exprs)
        assert result == ['no']

    def test_if_eq_combined(self):
        exprs = parse_sexpr("(&? (&= 1 1) yes no)")
        result = expand_macros(exprs)
        assert result == ['yes']

    def test_chr(self):
        exprs = parse_sexpr("((&chr 65))")
        result = expand_macros(exprs)
        assert result == [['A']]

    def test_ord(self):
        exprs = parse_sexpr("((&ord A))")
        result = expand_macros(exprs)
        assert result == [[65]]

    def test_nested_foreach(self):
        """Nested foreach-integer should produce cross product."""
        exprs = parse_sexpr(
            "(&foreach-integer I (1 2) (&foreach-integer J (1 2) (pair I J)))")
        result = expand_macros(exprs)
        assert result == [['pair', 1, 1], ['pair', 1, 2],
                          ['pair', 2, 1], ['pair', 2, 2]]

    def test_foreach_with_cat(self):
        """foreach-integer body containing &cat."""
        exprs = parse_sexpr(
            "(&foreach-integer I (1 3) (item (&. S I)))")
        result = expand_macros(exprs)
        assert result == [['item', 'S1'], ['item', 'S2'], ['item', 'S3']]

    def test_foreach_with_conditional_skip(self):
        """foreach with &? &= to conditionally emit empty."""
        exprs = parse_sexpr(
            "(&foreach-integer I (1 3) (&? (&= I 2) () (keep I)))")
        result = expand_macros(exprs)
        assert ['keep', 1] in result
        assert ['keep', 3] in result

    def test_foreach(self):
        """Basic &foreach over explicit list."""
        exprs = parse_sexpr("(&foreach X (a b c) (item X))")
        result = expand_macros(exprs)
        assert result == [['item', 'a'], ['item', 'b'], ['item', 'c']]

    def test_and_or_not(self):
        exprs = parse_sexpr("((&and 1 1) (&and 1 0) (&or 0 1) (&not 0))")
        result = expand_macros(exprs)
        assert result == [[1, 0, 1, 1]]

    def test_comparison_operators(self):
        exprs = parse_sexpr("((&> 3 2) (&< 3 2) (&>= 3 3) (&<= 2 3))")
        result = expand_macros(exprs)
        assert result == [[1, 0, 1, 1]]

    def test_scheme_raises(self):
        exprs = parse_sexpr("(&scheme (+ 1 2))")
        with pytest.raises(NotImplementedError, match="Guile Scheme"):
            expand_macros(exprs)

    def test_no_macros_passthrough(self):
        """Expressions without macros should pass through unchanged."""
        exprs = parse_sexpr("(grammar (name test) (transform (from (S)) (to ())))")
        result = expand_macros(exprs)
        assert result == exprs


# ---------------------------------------------------------------------------
# Parametric expression evaluator tests
# ---------------------------------------------------------------------------

class TestParametricExpressions:
    def test_numeric_literal(self):
        assert _eval_parametric(0.5, {}) == 0.5

    def test_parameter_lookup(self):
        assert _eval_parametric('stayProb', {'stayProb': 0.9}) == 0.9

    def test_infix_multiply(self):
        params = {'lambda': 1.5, 'pA': 0.25}
        assert _eval_parametric(['lambda', '*', 'pA'], params) == pytest.approx(0.375)

    def test_infix_divide(self):
        result = _eval_parametric(['leaveProb', '/', 9], {'leaveProb': 0.1})
        assert result == pytest.approx(0.1 / 9)

    def test_infix_chain(self):
        # lambda * (CLASS / CLASSES) * pA
        params = {'lambda': 1.0, 'pA': 0.25}
        expr = ['lambda', '*', 0.5, '*', 'pA']
        assert _eval_parametric(expr, params) == pytest.approx(0.125)

    def test_single_element_list(self):
        assert _eval_parametric(['stayProb'], {'stayProb': 0.9}) == 0.9

    def test_unknown_returns_none(self):
        assert _eval_parametric('unknown', {}) is None


# ---------------------------------------------------------------------------
# End-to-end macro grammar tests
# ---------------------------------------------------------------------------

class TestMacroGrammarParsing:
    def test_conservation_phylohmm(self):
        """Parse conservation_phylohmm.eg — 10-class parametric HMM."""
        text = open('/tmp/xrate_examples/conservation_phylohmm.eg').read()
        result = parse_xrate(text)

        # Should have 10 chains (one per rate class)
        assert len(result.chains) == 10

        # Chain terminals should be X1..X10
        chain_terminals = [c['terminals'][0] for c in result.chains]
        for i in range(1, 11):
            assert f'X{i}' in chain_terminals

        # All chains should be single-position
        for c in result.chains:
            assert c['n_positions'] == 1

        # Grammar should have nonterminals START, S1..S10, S1*..S10*
        g = result.grammar
        nt_names = [nt.name for nt in g.nonterminals]
        assert 'START' in nt_names
        for i in range(1, 11):
            assert f'S{i}' in nt_names
            assert f'S{i}*' in nt_names

        # Should classify as HMM
        assert classify_grammar(g) == 'hmm'
        validate_grammar(g)

    def test_conservation_phylohmm_chains_have_rates(self):
        """Verify expanded chains have non-trivial rate matrices."""
        text = open('/tmp/xrate_examples/conservation_phylohmm.eg').read()
        result = parse_xrate(text)

        # Rate class 10 (fastest) should have higher rates than class 1 (slowest)
        chain1 = next(c for c in result.chains if c['terminals'] == ['X1'])
        chain10 = next(c for c in result.chains if c['terminals'] == ['X10'])

        # Off-diagonal sum should be higher for class 10
        off_diag_1 = np.abs(chain1['rate_matrix']).sum() - np.abs(
            np.diag(chain1['rate_matrix'])).sum()
        off_diag_10 = np.abs(chain10['rate_matrix']).sum() - np.abs(
            np.diag(chain10['rate_matrix'])).sum()
        assert off_diag_10 > off_diag_1

    def test_overprinter_div(self):
        """Parse overprinter.eg — uses &div for 1/3 probability."""
        text = open('/tmp/xrate_examples/overprinter.eg').read()
        result = parse_xrate(text)

        # Should parse without errors
        assert len(result.chains) > 0
        assert len(result.grammar.rules) > 0

        # Find the chain with 1/3 initial probabilities
        found_third = False
        for chain in result.chains:
            for pi_val in chain['pi']:
                if abs(pi_val - 1/3) < 0.01:
                    found_third = True
                    break
        assert found_third, "Should find chain with 1/3 initial probability"

    def test_inline_macro_grammar(self):
        """Parse an inline grammar with &define and &foreach-integer."""
        text = """
        (&define N 3)
        (alphabet (name RNA) (token (a c g u)))
        (grammar
         (name test_macro)
         (&foreach-integer I (1 N)
          (chain
           (terminal ((&. X I)))
           (&foreach-token TOK
            (initial (state (TOK)) (prob (&/ 1 &TOKENS))))
           (&foreach-token SRC
            (&foreach-token DEST
             (&? (&= SRC DEST) ()
              (mutate (from (SRC)) (to (DEST)) (rate (&/ I N))))))))
         (&foreach-integer I (1 N)
          (transform (from (START)) (to ((&. S I))) (prob (&/ 1 N)))
          (transform (from ((&. S I))) (to ((&. X I) (&. S I *))))
          (transform (from ((&. S I *))) (to ()) (prob 1))))
        """
        result = parse_xrate(text)

        # 3 chains: X1, X2, X3
        assert len(result.chains) == 3
        assert result.chains[0]['terminals'] == ['X1']
        assert result.chains[2]['terminals'] == ['X3']

        # Each chain has uniform pi
        for c in result.chains:
            np.testing.assert_allclose(c['pi'], 0.25, atol=1e-10)

        # Rate matrices should scale: X3 has highest rates
        off_diag = []
        for c in result.chains:
            Q = c['rate_matrix']
            off_diag.append(Q[0, 1])
        assert off_diag[0] < off_diag[1] < off_diag[2]

        # Grammar should classify as HMM
        g = result.grammar
        assert classify_grammar(g) == 'hmm'
        validate_grammar(g)

    def test_xdecoder(self):
        """Parse XDecoder.eg — RNA structure in coding regions."""
        xdecoder_path = os.path.join(DATA_DIR, 'XDecoder.eg')
        result = parse_xrate_file(xdecoder_path)

        # 24 chains total: 12 non-structural + 3 loop + 9 paired
        assert len(result.chains) == 24
        single = [c for c in result.chains if c['n_positions'] == 1]
        paired = [c for c in result.chains if c['n_positions'] == 2]
        assert len(single) == 15
        assert len(paired) == 9

        # Grammar properties
        g = result.grammar
        assert len(g.nonterminals) == 86
        assert len(g.rules) == 149
        assert classify_grammar(g) == 'scfg'
        validate_grammar(g)

        # Should have structural nonterminals
        nt_names = [nt.name for nt in g.nonterminals]
        assert 'begin' in nt_names
        assert 'pfoldCodingS' in nt_names
        assert 'pfoldCodingB' in nt_names
