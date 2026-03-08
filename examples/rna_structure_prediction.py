"""RNA structure prediction using PFOLD-like SCFG.

Demonstrates:
1. Building an RNA SCFG grammar
2. Pre-computing terminal weights from a phylogenetic alignment
3. Running CYK/Viterbi to predict structure
"""

import jax.numpy as jnp
import numpy as np

from jaxrate import compile_grammar, TerminalWeights, viterbi
from jaxrate.presets import pfold_grammar


def main():
    # Build PFOLD-like grammar
    grammar = pfold_grammar()
    cg = compile_grammar(grammar)
    print(f"Grammar class: {cg.grammar_class}")
    print(f"Nonterminals: {[nt.name for nt in grammar.nonterminals]}")
    print(f"Rules: {len(grammar.rules)}")

    # Synthetic terminal weights for a short RNA
    # Columns: 0  1  2  3  4  5  6  7
    # Structure:  (  (  .  .  )  )  .  .
    C = 8

    # Model 0: unpaired emission weights
    unpaired = np.array([-2.0, -3.0, -1.0, -1.0, -3.0, -3.0, -1.0, -1.0])

    # Model 1: paired emission weights
    # Pairs (0,5) and (1,4) should have strong signal
    paired = np.full((C, C), -5.0)
    paired[0, 5] = -0.5  # strong pair
    paired[1, 4] = -0.5  # strong pair

    tw = TerminalWeights(
        single=jnp.array([unpaired, unpaired]),  # 2 models
        paired=jnp.array([paired, paired]),
        C=C,
    )

    # Run Viterbi
    labels, log_prob = viterbi(grammar, tw)
    print(f"\nViterbi log-prob: {log_prob:.4f}")
    print(f"Per-column labels: {labels}")

    # Map labels to nonterminal names
    nt_names = [nt.name for nt in grammar.nonterminals]
    annotation = [nt_names[int(l)] for l in labels]
    print(f"Annotation: {annotation}")


if __name__ == '__main__':
    main()
