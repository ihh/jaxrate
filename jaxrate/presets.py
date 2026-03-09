"""Pre-built grammars for common phylogenetic annotation tasks."""

import jax.numpy as jnp
import math

from .types import Grammar, Nonterminal, Rule, EmissionGroup
from .grammar import GrammarBuilder


def pfold_grammar():
    """PFOLD-like RNA structure SCFG.

    Nonterminals:
        S — start/structure (can be stem or loop)
        L — stem (paired emissions)
        F — loop/unpaired region

    Rules:
        S → L S        (stem followed by more structure)
        S → F          (loop region)
        L → e₁ S e₂   (paired emission flanking inner structure)
        F → e F        (emit unpaired, continue loop)
        F → e          (emit unpaired, end)

    Models:
        0 — unpaired emission model
        1 — paired emission model

    Returns:
        Grammar
    """
    gb = GrammarBuilder()

    S = gb.add_nonterminal('S')
    L = gb.add_nonterminal('L')
    F = gb.add_nonterminal('F')

    # S → L S  (stem then more)
    gb.add_rule(S, rhs=[L, S], log_weight=jnp.log(0.4))

    # S → F  (just a loop)
    gb.add_rule(S, rhs=[F], log_weight=jnp.log(0.6))

    # L → e₁ S e₂  (paired emission flanking S)
    gb.add_rule(L, rhs=[S],
                emissions=[EmissionGroup(n_positions=2, model_index=1)],
                log_weight=0.0)

    # F → e F  (emit unpaired, continue)
    gb.add_rule(F, rhs=[F],
                emissions=[EmissionGroup(n_positions=1, model_index=0)],
                log_weight=jnp.log(0.6))

    # F → e  (emit unpaired, end)
    gb.add_rule(F, rhs=[],
                emissions=[EmissionGroup(n_positions=1, model_index=0)],
                log_weight=jnp.log(0.4))

    return gb.build(start=S, n_models=2)


def gene_finder_grammar():
    """Gene-finding HMM.

    States:
        IG — intergenic (single nucleotide emission)
        E1, E2, E3 — exon codon positions 1, 2, 3
        I — intron (single nucleotide emission)

    Transitions model a simplified gene structure:
        IG → IG | E1
        E1 → E2
        E2 → E3
        E3 → E1 | I | IG
        I → I | E1

    Models:
        0 — intergenic model (single nucleotide)
        1 — exon model (single nucleotide, codon-position-specific)
        2 — intron model (single nucleotide)

    For codon-level phylogenetic models, the caller should pre-compute
    terminal weights using kmer_tokenize with stride=1 for codon windows.

    Returns:
        Grammar
    """
    gb = GrammarBuilder()

    IG = gb.add_nonterminal('IG')
    E1 = gb.add_nonterminal('E1')
    E2 = gb.add_nonterminal('E2')
    E3 = gb.add_nonterminal('E3')
    I = gb.add_nonterminal('I')

    p_gene = 0.01      # probability of starting a gene
    p_stop = 0.01      # probability of stopping a gene
    p_intron = 0.02    # probability of entering intron
    p_exon_resume = 0.05  # probability of leaving intron

    # IG → emit IG
    gb.add_rule(IG, rhs=[IG],
                emissions=[EmissionGroup(1, 0)],
                log_weight=math.log(1 - p_gene))
    # IG → emit E1
    gb.add_rule(IG, rhs=[E1],
                emissions=[EmissionGroup(1, 0)],
                log_weight=math.log(p_gene))
    # IG → emit (terminal)
    gb.add_rule(IG, rhs=[],
                emissions=[EmissionGroup(1, 0)],
                log_weight=math.log(1e-10))  # very small term prob

    # E1 → emit E2
    gb.add_rule(E1, rhs=[E2],
                emissions=[EmissionGroup(1, 1)],
                log_weight=0.0)

    # E2 → emit E3
    gb.add_rule(E2, rhs=[E3],
                emissions=[EmissionGroup(1, 1)],
                log_weight=0.0)

    # E3 → emit E1 (continue codon)
    gb.add_rule(E3, rhs=[E1],
                emissions=[EmissionGroup(1, 1)],
                log_weight=math.log(1 - p_stop - p_intron))
    # E3 → emit I (enter intron)
    gb.add_rule(E3, rhs=[I],
                emissions=[EmissionGroup(1, 1)],
                log_weight=math.log(p_intron))
    # E3 → emit IG (stop gene)
    gb.add_rule(E3, rhs=[IG],
                emissions=[EmissionGroup(1, 1)],
                log_weight=math.log(p_stop))
    # E3 → emit (terminal)
    gb.add_rule(E3, rhs=[],
                emissions=[EmissionGroup(1, 1)],
                log_weight=math.log(1e-10))

    # I → emit I
    gb.add_rule(I, rhs=[I],
                emissions=[EmissionGroup(1, 2)],
                log_weight=math.log(1 - p_exon_resume))
    # I → emit E1 (resume exon)
    gb.add_rule(I, rhs=[E1],
                emissions=[EmissionGroup(1, 2)],
                log_weight=math.log(p_exon_resume))

    return gb.build(start=IG, n_models=3)


def pseudoknot_grammar(max_span=None):
    """Pseudoknot-capable MCFG with fan-out 2.

    Nonterminals:
        S — start (fan-out 1): generates full sequence
        PK — pseudoknot (fan-out 2): generates two crossing stem-loop regions
        L — stem (fan-out 1): generates a stem-loop

    Rules:
        S → L S       (stem-loop followed by more structure)
        S → PK S      (pseudoknot followed by more structure)
        S → e         (single unpaired emission)
        L → e₁ S e₂   (paired emission flanking inner structure)
        PK(x, y) → L(x) L(y)  (two crossing stems)

    Models:
        0 — unpaired emission model
        1 — paired emission model

    Args:
        max_span: Maximum span per component for the PK nonterminal.
                  None means unlimited. Constraining this reduces complexity
                  from O(C⁶K³) to O(C²L²K³ + C³K³).

    Returns:
        Grammar
    """
    gb = GrammarBuilder()

    S = gb.add_nonterminal('S', fan_out=1)
    PK = gb.add_nonterminal('PK', fan_out=2, max_span=max_span)
    L = gb.add_nonterminal('L', fan_out=1)

    # S → L S  (stem then more structure)
    gb.add_rule(S, rhs=[L, S], log_weight=jnp.log(0.2))

    # S → e S  (unpaired then more structure)
    gb.add_rule(S, rhs=[S],
                emissions=[EmissionGroup(1, 0)],
                log_weight=jnp.log(0.3))

    # S → PK  (pseudoknot region, concatenates PK's two components)
    gb.add_rule(S, rhs=[PK], log_weight=jnp.log(0.1))

    # S → e (terminal unpaired)
    gb.add_rule(S, rhs=[],
                emissions=[EmissionGroup(1, 0)],
                log_weight=jnp.log(0.4))

    # L → e₁ S e₂ (paired emission flanking inner structure)
    gb.add_rule(L, rhs=[S],
                emissions=[EmissionGroup(2, 1)],
                log_weight=0.0)

    # PK(ax, by) → PK(x, y) [pair(a,b)]  (extend stem 1: left-left)
    # Pairs left of component 1 with left of component 2
    gb.add_rule(PK, rhs=[PK],
                emissions=[EmissionGroup(2, 1)],
                composition=('ll',),
                log_weight=jnp.log(0.4))

    # PK(xa, yb) → PK(x, y) [pair(a,b)]  (extend stem 2: right-right)
    # Pairs right of component 1 with right of component 2
    gb.add_rule(PK, rhs=[PK],
                emissions=[EmissionGroup(2, 1)],
                composition=('rr',),
                log_weight=jnp.log(0.4))

    # PK(x, y) → S(x) S(y)  (base: each component is independent structure)
    gb.add_rule(PK, rhs=[S, S], log_weight=jnp.log(0.2))

    return gb.build(start=S, n_models=2)
