"""Gene finding using HMM with phylogenetic terminal weights.

Demonstrates:
1. Building a gene-finder HMM
2. Viterbi decoding to annotate coding vs. non-coding regions
3. Forward-backward posteriors
"""

import jax.numpy as jnp
import numpy as np
import math

from jaxrate import compile_grammar, TerminalWeights, inside, viterbi
from jaxrate.hmm import hmm_posteriors
from jaxrate.presets import gene_finder_grammar


def main():
    grammar = gene_finder_grammar()
    cg = compile_grammar(grammar)
    print(f"Grammar class: {cg.grammar_class}")
    print(f"States: {[nt.name for nt in grammar.nonterminals]}")

    # Synthetic: 30 columns, intergenic(10) + exon(12) + intergenic(8)
    C = 30
    np.random.seed(42)

    # Model 0: intergenic (uniform moderate)
    ig_ll = np.full(C, -2.0)

    # Model 1: exon (strong signal in coding region)
    exon_ll = np.full(C, -3.0)  # weak by default
    exon_ll[10:22] = -0.5       # strong in coding region

    # Model 2: intron
    intron_ll = np.full(C, -2.5)

    tw = TerminalWeights(
        single=jnp.array([ig_ll, exon_ll, intron_ll]),
        C=C,
    )

    # Log-likelihood
    _, ll = inside(grammar, tw)
    print(f"\nLog-likelihood: {ll:.4f}")

    # Viterbi decoding
    labels, log_prob = viterbi(grammar, tw)
    nt_names = [nt.name for nt in grammar.nonterminals]
    annotation = [nt_names[int(l)] for l in labels]

    print(f"\nViterbi log-prob: {log_prob:.4f}")
    print(f"Annotation:")
    for i in range(0, C, 10):
        cols = list(range(i, min(i + 10, C)))
        ann = [annotation[c][:2] for c in cols]
        print(f"  Columns {i:2d}-{cols[-1]:2d}: {' '.join(f'{a:>3s}' for a in ann)}")

    # Posteriors
    posteriors, _ = hmm_posteriors(cg, tw)
    print(f"\nState posteriors shape: {posteriors.shape}")
    print(f"Sum check (should be 1.0): {float(posteriors[0].sum()):.6f}")


if __name__ == '__main__':
    main()
