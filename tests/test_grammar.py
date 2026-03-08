"""Tests for grammar construction, validation, and compilation."""

import pytest
import jax.numpy as jnp
import math

from jaxrate import (
    Grammar, Nonterminal, Rule, EmissionGroup,
    GrammarBuilder, compile_grammar, classify_grammar, validate_grammar,
)
from jaxrate.presets import pfold_grammar, gene_finder_grammar, pseudoknot_grammar


class TestGrammarBuilder:
    def test_basic_build(self):
        gb = GrammarBuilder()
        S = gb.add_nonterminal('S')
        gb.add_rule(S, rhs=[], emissions=[EmissionGroup(1, 0)],
                    log_weight=0.0)
        grammar = gb.build(start=S, n_models=1)

        assert len(grammar.nonterminals) == 1
        assert len(grammar.rules) == 1
        assert grammar.start == 0

    def test_multiple_nonterminals(self):
        gb = GrammarBuilder()
        S = gb.add_nonterminal('S')
        A = gb.add_nonterminal('A')
        assert S == 0
        assert A == 1


class TestClassify:
    def test_hmm(self):
        gb = GrammarBuilder()
        S = gb.add_nonterminal('S')
        gb.add_rule(S, rhs=[S], emissions=[EmissionGroup(1, 0)],
                    log_weight=math.log(0.8))
        gb.add_rule(S, rhs=[], emissions=[EmissionGroup(1, 0)],
                    log_weight=math.log(0.2))
        grammar = gb.build(n_models=1)
        assert classify_grammar(grammar) == 'hmm'

    def test_scfg(self):
        grammar = pfold_grammar()
        assert classify_grammar(grammar) == 'scfg'

    def test_mcfg(self):
        grammar = pseudoknot_grammar()
        assert classify_grammar(grammar) == 'mcfg'


class TestValidation:
    def test_valid_grammar(self):
        grammar = pfold_grammar()
        validate_grammar(grammar)  # should not raise

    def test_empty_grammar(self):
        grammar = Grammar(nonterminals=[], rules=[], start=0, n_models=1)
        with pytest.raises(ValueError, match="at least one nonterminal"):
            validate_grammar(grammar)

    def test_bad_start_index(self):
        grammar = Grammar(
            nonterminals=[Nonterminal('S')],
            rules=[],
            start=5,
            n_models=1,
        )
        with pytest.raises(ValueError, match="out of range"):
            validate_grammar(grammar)

    def test_bad_model_index(self):
        grammar = Grammar(
            nonterminals=[Nonterminal('S')],
            rules=[Rule(0, (), (EmissionGroup(1, 5),), 0.0)],
            start=0,
            n_models=1,
        )
        with pytest.raises(ValueError, match="model_index"):
            validate_grammar(grammar)


class TestCompile:
    def test_compile_hmm(self):
        grammar = gene_finder_grammar()
        cg = compile_grammar(grammar)
        assert cg.grammar_class == 'hmm'
        assert cg.n_nonterminals == 5
        assert cg.n_rules == len(grammar.rules)

    def test_compile_scfg(self):
        grammar = pfold_grammar()
        cg = compile_grammar(grammar)
        assert cg.grammar_class == 'scfg'

    def test_compile_mcfg(self):
        grammar = pseudoknot_grammar()
        cg = compile_grammar(grammar)
        assert cg.grammar_class == 'mcfg'

    def test_rules_for_nt(self):
        grammar = pfold_grammar()
        cg = compile_grammar(grammar)
        # Each nonterminal should have at least one rule
        for nt in range(cg.n_nonterminals):
            assert int(cg.n_rules_for_nt[nt]) > 0


class TestPresets:
    def test_pfold_valid(self):
        grammar = pfold_grammar()
        validate_grammar(grammar)
        assert grammar.n_models == 2

    def test_gene_finder_valid(self):
        grammar = gene_finder_grammar()
        validate_grammar(grammar)
        assert grammar.n_models == 3
        assert len(grammar.nonterminals) == 5

    def test_pseudoknot_valid(self):
        grammar = pseudoknot_grammar()
        validate_grammar(grammar)
        assert grammar.n_models == 2
        # Should have fan-out 2 nonterminal
        fan_outs = [nt.fan_out for nt in grammar.nonterminals]
        assert 2 in fan_outs
