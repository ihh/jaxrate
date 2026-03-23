/**
 * jaxrate-web: stochastic MCFGs with phylogenetic terminal weights.
 *
 * Phase 1: CPU implementations (JAX parity via TypeScript)
 * Phase 2: WebGPU-accelerated HMM (diagonal sweep)
 */

// ── Types ───────────────────────────────────────────────────────────

export type {
  Nonterminal, EmissionGroup, Rule, Grammar,
  CompiledGrammar, TerminalWeights, ParseResult,
  GrammarClass,
} from './types.js';

// ── Log-semiring ────────────────────────────────────────────────────

export { NEG_INF, logaddexp, logsumexp, logMatvec, logMatvecT, negInfArray } from './log-semiring.js';

// ── Grammar ─────────────────────────────────────────────────────────

export {
  GrammarBuilder,
  classifyGrammar, validateGrammar, compileGrammar,
  classifyRules,
} from './grammar.js';

// ── CPU algorithms ──────────────────────────────────────────────────

// HMM
export {
  buildHmmTables,
  hmmForward, hmmBackward, hmmViterbi, hmmPosteriors,
} from './hmm.js';
export type { HmmTables, ForwardResult, BackwardResult, PosteriorResult } from './hmm.js';

// SCFG
export {
  scfgInside, scfgOutside, scfgViterbi, scfgPosteriors,
} from './scfg.js';
export type { InsideResult, OutsideResult, ScfgPosteriorResult } from './scfg.js';

// MCFG
export { mcfgInside, mcfgViterbi } from './mcfg.js';
export type { McfgInsideResult } from './mcfg.js';

// ── Presets ──────────────────────────────────────────────────────────

export { pfoldGrammar, geneFinderGrammar, pseudoknotGrammar } from './presets.js';

// ── xrate parser ────────────────────────────────────────────────────

export { parseSExpr, parseXrate, parseXrateFile } from './xrate-parser.js';
export type { XrateGrammar, ChainModel, SExpr } from './xrate-parser.js';

// ── Neighbor-Joining ────────────────────────────────────────────────

export { neighborJoining, hammingDistances, jukesCantor } from './nj.js';
export type { NJResult } from './nj.js';

// ── WebGPU (Phase 2) ────────────────────────────────────────────────

export {
  getDevice, releaseDevice,
  hmmForwardGpu, hmmBackwardGpu, hmmViterbiGpu, hmmPosteriorsGpu,
  scfgInsideGpu, scfgViterbiGpu,
} from './webgpu/index.js';

// ── High-level dispatchers ──────────────────────────────────────────

import type { Grammar, TerminalWeights, ParseResult } from './types.js';
import { compileGrammar } from './grammar.js';
import { hmmForward, hmmViterbi } from './hmm.js';
import { scfgInside, scfgViterbi } from './scfg.js';
import { mcfgInside, mcfgViterbi } from './mcfg.js';

/** Auto-dispatch inside algorithm by grammar class. */
export function inside(grammar: Grammar, tw: TerminalWeights) {
  const cg = compileGrammar(grammar);
  switch (cg.grammarClass) {
    case 'hmm': return hmmForward(cg, tw);
    case 'scfg': return scfgInside(cg, tw);
    case 'mcfg': return mcfgInside(cg, tw);
  }
}

/** Auto-dispatch Viterbi decoding by grammar class. */
export function viterbi(grammar: Grammar, tw: TerminalWeights): ParseResult {
  const cg = compileGrammar(grammar);
  switch (cg.grammarClass) {
    case 'hmm': return hmmViterbi(cg, tw);
    case 'scfg': return scfgViterbi(cg, tw);
    case 'mcfg': return mcfgViterbi(cg, tw);
  }
}
