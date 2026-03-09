"""Core data structures for jaxrate grammars.

All types are NamedTuples for JAX pytree compatibility.
"""

from typing import NamedTuple, Optional
import jax.numpy as jnp


class Nonterminal(NamedTuple):
    """A nonterminal symbol in the grammar.

    Attributes:
        name: Human-readable name (e.g., 'S', 'Stem', 'Loop')
        fan_out: Number of string components (1 for HMM/SCFG, 2 for MCFG)
        max_span: Maximum span per component (None = unlimited).
                  For fan-out 2, constrains each component independently.
    """
    name: str
    fan_out: int = 1
    max_span: Optional[int] = None


class EmissionGroup(NamedTuple):
    """A group of positions emitted jointly under one model.

    Attributes:
        n_positions: Number of alignment columns emitted (e.g., 1 for unpaired,
                     2 for base pair, 3 for codon)
        model_index: Index into the terminal weights table
    """
    n_positions: int
    model_index: int


class Rule(NamedTuple):
    """A production rule in the grammar.

    Attributes:
        lhs: Index of the left-hand side nonterminal
        rhs: Tuple of indices of right-hand side nonterminals (empty for terminal rules)
        emissions: Tuple of EmissionGroup describing emitted columns
        log_weight: Log probability of this rule
        composition: Per-LHS-component list of items describing how RHS components
                     and emissions compose to form the LHS string.
                     Each item is ('nt', rhs_index, component_index) or
                     ('emit', group_index, position_index).
    """
    lhs: int
    rhs: tuple
    emissions: tuple
    log_weight: float
    composition: tuple = ()


class Grammar(NamedTuple):
    """A stochastic grammar for phylogenetic annotation.

    Attributes:
        nonterminals: List of Nonterminal objects
        rules: List of Rule objects
        start: Index of the start nonterminal
        n_models: Number of distinct emission models
    """
    nonterminals: list
    rules: list
    start: int = 0
    n_models: int = 1


class CompiledGrammar(NamedTuple):
    """JIT-friendly compiled grammar with dense arrays.

    Attributes:
        grammar_class: 'hmm', 'scfg', or 'mcfg'
        n_nonterminals: Number of nonterminals
        n_rules: Number of rules
        rule_lhs: (n_rules,) int32 — LHS nonterminal index for each rule
        rule_rhs: (n_rules, max_rhs) int32 — RHS nonterminal indices (-1 for padding)
        rule_n_rhs: (n_rules,) int32 — number of RHS symbols per rule
        rule_log_weights: (n_rules,) float64 — log-weights
        rule_emission_model: (n_rules, max_emit) int32 — model index per emission group (-1 pad)
        rule_emission_npos: (n_rules, max_emit) int32 — n_positions per emission group (0 pad)
        rule_n_emissions: (n_rules,) int32 — number of emission groups per rule
        rules_for_nt: (n_nonterminals, max_rules_per_nt) int32 — rule indices per NT (-1 pad)
        n_rules_for_nt: (n_nonterminals,) int32 — count of rules per NT
        start: int — start nonterminal index
        n_models: int — number of emission models
        fan_outs: (n_nonterminals,) int32 — fan-out per nonterminal
        max_spans: (n_nonterminals,) int32 — max span per component (-1 = unlimited)
        rule_composition: (n_rules,) int32 — composition type per rule:
            0 = default, 1 = 'll' (left-left paired), 2 = 'rr' (right-right paired)
    """
    grammar_class: str
    n_nonterminals: int
    n_rules: int
    rule_lhs: jnp.ndarray
    rule_rhs: jnp.ndarray
    rule_n_rhs: jnp.ndarray
    rule_log_weights: jnp.ndarray
    rule_emission_model: jnp.ndarray
    rule_emission_npos: jnp.ndarray
    rule_n_emissions: jnp.ndarray
    rules_for_nt: jnp.ndarray
    n_rules_for_nt: jnp.ndarray
    start: int
    n_models: int
    fan_outs: jnp.ndarray
    max_spans: jnp.ndarray = None
    rule_composition: jnp.ndarray = None


class TerminalWeights(NamedTuple):
    """Pre-computed phylogenetic terminal weights.

    Attributes:
        single: (K_single, C) float64 — log-likelihoods for single-column models
        paired: (K_paired, C, C) float64 — log-likelihoods for column pairs
                (None if no paired models)
        kmer: (K_kmer, C_kmer) float64 — log-likelihoods for k-mer windows
              (None if no k-mer models)
        C: int — number of alignment columns
    """
    single: jnp.ndarray
    paired: Optional[jnp.ndarray] = None
    kmer: Optional[jnp.ndarray] = None
    C: int = 0


class ParseTree(NamedTuple):
    """Result of Viterbi/CYK decoding.

    Attributes:
        labels: (C,) int32 — per-column nonterminal or state labels
        log_prob: float — log-probability of the best parse
        rule_trace: optional trace of rules used in derivation
    """
    labels: jnp.ndarray
    log_prob: float
    rule_trace: Optional[list] = None


class TrainState(NamedTuple):
    """State of EM training.

    Attributes:
        log_weights: (n_rules,) float64 — current rule log-weights
        iteration: int — current iteration number
        log_likelihood: float — current log-likelihood
    """
    log_weights: jnp.ndarray
    iteration: int = 0
    log_likelihood: float = float('-inf')
