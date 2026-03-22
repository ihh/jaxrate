/**
 * Core types for jaxrate-web.
 * Mirrors jaxrate/types.py — plain objects instead of NamedTuples.
 */

/** Nonterminal symbol in a grammar. */
export interface Nonterminal {
  name: string;
  fanOut: number;    // 1 for HMM/SCFG, 2 for MCFG
  maxSpan: number;   // -1 = unlimited
}

/** Group of alignment columns emitted jointly by a rule. */
export interface EmissionGroup {
  nPositions: number;  // 1=single, 2=paired, 3=codon
  modelIndex: number;  // index into terminal weights
}

/** Production rule in a grammar. */
export interface Rule {
  lhs: number;                 // LHS nonterminal index
  rhs: number[];               // RHS nonterminal indices
  emissions: EmissionGroup[];  // emitted column groups
  logWeight: number;           // log probability
  composition: number;         // 0=default, 1='ll', 2='rr'
}

/** Grammar definition. */
export interface Grammar {
  nonterminals: Nonterminal[];
  rules: Rule[];
  start: number;     // start nonterminal index
  nModels: number;   // number of distinct emission models
}

/** Dense compiled grammar for fast algorithm execution. */
export interface CompiledGrammar {
  grammarClass: 'hmm' | 'scfg' | 'mcfg';
  K: number;              // number of nonterminals
  nRules: number;         // total number of rules

  // Per-rule arrays (length nRules)
  ruleLhs: Int32Array;
  ruleRhs: Int32Array[];       // ruleRhs[r] = int32[] of RHS NT indices
  ruleNRhs: Int32Array;
  ruleLogWeights: Float64Array;
  ruleEmissionModel: Int32Array[];  // ruleEmissionModel[r] = model indices
  ruleEmissionNPos: Int32Array[];   // ruleEmissionNPos[r] = n_positions
  ruleNEmissions: Int32Array;
  ruleComposition: Int32Array;

  // Per-nonterminal: which rules have this NT as LHS
  rulesForNt: Int32Array[];    // rulesForNt[k] = rule indices
  fanOuts: Int32Array;
  maxSpans: Int32Array;        // -1 = unlimited
}

/** Terminal weights: log-likelihoods from phylogenetic models. */
export interface TerminalWeights {
  /** (K_single, C) — single[model][col] */
  single: Float64Array[];
  /** (K_paired, C, C) — paired[model][i * C + j], or null */
  paired: Float64Array[] | null;
  /** (K_kmer, C_kmer) — kmer[model][window], or null */
  kmer: Float64Array[] | null;
  C: number;  // number of alignment columns
}

/** Result of Viterbi/CYK decoding. */
export interface ParseResult {
  labels: Int32Array;   // (C,) per-column nonterminal labels
  logProb: number;      // best parse probability
}

/** Grammar classification. */
export type GrammarClass = 'hmm' | 'scfg' | 'mcfg';
