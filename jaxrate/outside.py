"""General outside algorithm — dispatches by grammar class."""

from .grammar import compile_grammar
from .hmm import hmm_backward
from .scfg import scfg_outside
from .types import CompiledGrammar


def outside(grammar, terminal_weights, inside_chart=None):
    """Compute outside (backward) probabilities.

    Args:
        grammar: Grammar or CompiledGrammar
        terminal_weights: TerminalWeights
        inside_chart: pre-computed inside chart (optional; computed if None)

    Returns:
        chart: algorithm-specific outside chart
        log_likelihood: scalar
    """
    if isinstance(grammar, CompiledGrammar):
        cg = grammar
    else:
        cg = compile_grammar(grammar)

    if cg.grammar_class == 'hmm':
        beta, ll = hmm_backward(cg, terminal_weights)
        return beta, ll
    elif cg.grammar_class == 'scfg':
        if inside_chart is None:
            from .scfg import scfg_inside
            inside_chart, _ = scfg_inside(cg, terminal_weights)
        beta = scfg_outside(cg, terminal_weights, inside_chart)
        ll = inside_chart[cg.start, 0, terminal_weights.C]
        return beta, ll
    else:
        raise NotImplementedError("MCFG outside not yet implemented")
