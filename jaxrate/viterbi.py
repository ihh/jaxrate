"""General Viterbi/CYK decoding — dispatches by grammar class."""

from .grammar import compile_grammar
from .hmm import hmm_viterbi
from .scfg import scfg_viterbi
from .mcfg import mcfg_viterbi
from .types import CompiledGrammar


def viterbi(grammar, terminal_weights):
    """Find the most probable parse/labeling.

    Dispatches to the appropriate algorithm based on grammar class.

    Args:
        grammar: Grammar or CompiledGrammar
        terminal_weights: TerminalWeights

    Returns:
        labels: (C,) int32 — per-column state/nonterminal labels
        log_prob: scalar — log probability of best parse
    """
    if isinstance(grammar, CompiledGrammar):
        cg = grammar
    else:
        cg = compile_grammar(grammar)

    if cg.grammar_class == 'hmm':
        labels, log_prob = hmm_viterbi(cg, terminal_weights)
        return labels, log_prob
    elif cg.grammar_class == 'scfg':
        labels, log_prob, _ = scfg_viterbi(cg, terminal_weights)
        return labels, log_prob
    else:
        labels, log_prob = mcfg_viterbi(cg, terminal_weights)
        return labels, log_prob
