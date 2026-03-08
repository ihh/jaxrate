"""General inside algorithm — dispatches by grammar class."""

from .grammar import compile_grammar
from .hmm import hmm_forward
from .scfg import scfg_inside
from .mcfg import mcfg_inside
from .types import CompiledGrammar


def inside(grammar, terminal_weights):
    """Compute inside (forward) probabilities.

    Dispatches to the appropriate algorithm based on grammar class:
    - 'hmm': scan-based forward algorithm O(CK²)
    - 'scfg': chart-based inside algorithm O(C³K³)
    - 'mcfg': fan-out 2 chart algorithm O(C⁶K³)

    Args:
        grammar: Grammar or CompiledGrammar
        terminal_weights: TerminalWeights

    Returns:
        chart: algorithm-specific chart structure
        log_likelihood: scalar log P(x)
    """
    if isinstance(grammar, CompiledGrammar):
        cg = grammar
    else:
        cg = compile_grammar(grammar)

    if cg.grammar_class == 'hmm':
        alpha, ll = hmm_forward(cg, terminal_weights)
        return alpha, ll
    elif cg.grammar_class == 'scfg':
        alpha, ll = scfg_inside(cg, terminal_weights)
        return alpha, ll
    else:
        alpha1, alpha2, ll = mcfg_inside(cg, terminal_weights)
        return (alpha1, alpha2), ll
