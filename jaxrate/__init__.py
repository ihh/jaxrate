"""jaxrate: Stochastic MCFGs with phylogenetic terminal weights.

Named after xrate. Implements HMMs, SCFGs, and MCFGs for phylogenetic
sequence annotation using JAX.
"""

from .types import (
    Nonterminal,
    EmissionGroup,
    Rule,
    Grammar,
    CompiledGrammar,
    TerminalWeights,
    ParseTree,
    TrainState,
)

from .grammar import (
    compile_grammar,
    classify_grammar,
    validate_grammar,
    GrammarBuilder,
)

from .inside import inside
from .outside import outside
from .viterbi import viterbi

from .terminal_weights import precompute_terminal_weights

from .train import train, em_step

from .simulate import simulate_parse

from .presets import (
    pfold_grammar,
    gene_finder_grammar,
    pseudoknot_grammar,
)

__version__ = "0.1.0"
