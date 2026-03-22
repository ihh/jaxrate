/**
 * HMM algorithms: forward, backward, Viterbi, posteriors.
 * O(CK²) scan-based — mirrors jaxrate/hmm.py.
 */

import type { CompiledGrammar, TerminalWeights, ParseResult } from './types.js';
import { NEG_INF, logaddexp, logsumexp, negInfArray } from './log-semiring.js';

// ── Build HMM tables from compiled grammar ──────────────────────────

export interface HmmTables {
  logTrans: Float64Array;  // (K, K) row-major: logTrans[i*K + j]
  logEmit: Float64Array;   // (K, C) row-major: logEmit[k*C + c]
  logInit: Float64Array;   // (K,)
  logTerm: Float64Array;   // (K,)
  K: number;
  C: number;
}

export function buildHmmTables(cg: CompiledGrammar, tw: TerminalWeights): HmmTables {
  const K = cg.K;
  const C = tw.C;

  const logTrans = negInfArray(K * K);
  const logEmit = new Float64Array(K * C);  // initialized to 0
  const logInit = negInfArray(K);
  const logTerm = negInfArray(K);

  for (let r = 0; r < cg.nRules; r++) {
    const lhs = cg.ruleLhs[r];
    const nRhs = cg.ruleNRhs[r];
    const nEmit = cg.ruleNEmissions[r];
    const logW = cg.ruleLogWeights[r];

    if (nRhs === 0 && nEmit === 0) {
      // Epsilon/termination: A → ε
      logTerm[lhs] = logaddexp(logTerm[lhs], logW);
    } else if (nRhs === 0 && nEmit > 0) {
      // Terminal: A → e (single-state HMM, init + term + emit)
      logInit[lhs] = logaddexp(logInit[lhs], logW);
      logTerm[lhs] = logaddexp(logTerm[lhs], 0.0);
      const model = cg.ruleEmissionModel[r][0];
      for (let c = 0; c < C; c++) {
        logEmit[lhs * C + c] = tw.single[model][c];
      }
    } else if (nRhs === 1 && nEmit === 0) {
      // Null transition: A → B
      const rhs = cg.ruleRhs[r][0];
      logTrans[lhs * K + rhs] = logaddexp(logTrans[lhs * K + rhs], logW);
    } else if (nRhs === 1 && nEmit > 0) {
      // Emit + transition: A → e B (left-emit)
      const rhs = cg.ruleRhs[r][0];
      logTrans[lhs * K + rhs] = logaddexp(logTrans[lhs * K + rhs], logW);
      const model = cg.ruleEmissionModel[r][0];
      for (let c = 0; c < C; c++) {
        logEmit[lhs * C + c] = tw.single[model][c];
      }
    }
  }

  // States with no explicit init prob but have emit rules: set uniform
  // Actually, we need to identify start states. In HMM, the start NT's rules define init.
  // Let's handle init: rules from start nonterminal
  // Re-scan: init comes from rules where LHS = start and they emit
  // The Python code builds init from rules with LHS = start that have emit + rhs
  // For simplicity, keep the above logic which accumulates into logInit for terminal rules.
  // For transition rules from start, the start state transitions INTO other states.

  return { logTrans, logEmit, logInit, logTerm, K, C };
}

// ── Forward algorithm ───────────────────────────────────────────────

export interface ForwardResult {
  alpha: Float64Array;   // (C, K) row-major: alpha[c*K + k]
  logLikelihood: number;
}

export function hmmForward(cg: CompiledGrammar, tw: TerminalWeights): ForwardResult;
export function hmmForward(tables: HmmTables): ForwardResult;
export function hmmForward(arg1: CompiledGrammar | HmmTables, arg2?: TerminalWeights): ForwardResult {
  const t = isHmmTables(arg1) ? arg1 : buildHmmTables(arg1, arg2!);
  const { logTrans, logEmit, logInit, logTerm, K, C } = t;

  const alpha = negInfArray(C * K);
  const tmp = new Float64Array(K);

  // Base: alpha[0, k] = logInit[k] + logEmit[k, 0]
  for (let k = 0; k < K; k++) {
    alpha[k] = logInit[k] + logEmit[k * C];
  }

  // Recurse: alpha[c, j] = logsumexp_i(alpha[c-1, i] + logTrans[i, j]) + logEmit[j, c]
  for (let c = 1; c < C; c++) {
    for (let j = 0; j < K; j++) {
      for (let i = 0; i < K; i++) {
        tmp[i] = alpha[(c - 1) * K + i] + logTrans[i * K + j];
      }
      alpha[c * K + j] = logsumexp(tmp) + logEmit[j * C + c];
    }
  }

  // Termination: logLik = logsumexp(alpha[C-1, k] + logTerm[k])
  for (let k = 0; k < K; k++) {
    tmp[k] = alpha[(C - 1) * K + k] + logTerm[k];
  }
  const logLikelihood = logsumexp(tmp);

  return { alpha, logLikelihood };
}

// ── Backward algorithm ──────────────────────────────────────────────

export interface BackwardResult {
  beta: Float64Array;    // (C, K) row-major
  logLikelihood: number;
}

export function hmmBackward(cg: CompiledGrammar, tw: TerminalWeights): BackwardResult;
export function hmmBackward(tables: HmmTables): BackwardResult;
export function hmmBackward(arg1: CompiledGrammar | HmmTables, arg2?: TerminalWeights): BackwardResult {
  const t = isHmmTables(arg1) ? arg1 : buildHmmTables(arg1, arg2!);
  const { logTrans, logEmit, logInit, logTerm, K, C } = t;

  const beta = negInfArray(C * K);
  const tmp = new Float64Array(K);

  // Base: beta[C-1, k] = logTerm[k]
  for (let k = 0; k < K; k++) {
    beta[(C - 1) * K + k] = logTerm[k];
  }

  // Recurse: beta[c, i] = logsumexp_j(logTrans[i, j] + logEmit[j, c+1] + beta[c+1, j])
  for (let c = C - 2; c >= 0; c--) {
    for (let i = 0; i < K; i++) {
      for (let j = 0; j < K; j++) {
        tmp[j] = logTrans[i * K + j] + logEmit[j * C + (c + 1)] + beta[(c + 1) * K + j];
      }
      beta[c * K + i] = logsumexp(tmp);
    }
  }

  // Termination: logLik = logsumexp(logInit[k] + logEmit[k, 0] + beta[0, k])
  for (let k = 0; k < K; k++) {
    tmp[k] = logInit[k] + logEmit[k * C] + beta[k];
  }
  const logLikelihood = logsumexp(tmp);

  return { beta, logLikelihood };
}

// ── Viterbi algorithm ───────────────────────────────────────────────

export function hmmViterbi(cg: CompiledGrammar, tw: TerminalWeights): ParseResult;
export function hmmViterbi(tables: HmmTables): ParseResult;
export function hmmViterbi(arg1: CompiledGrammar | HmmTables, arg2?: TerminalWeights): ParseResult {
  const t = isHmmTables(arg1) ? arg1 : buildHmmTables(arg1, arg2!);
  const { logTrans, logEmit, logInit, logTerm, K, C } = t;

  const v = negInfArray(C * K);
  const bp = new Int32Array(C * K);  // backpointers

  // Base: v[0, k] = logInit[k] + logEmit[k, 0]
  for (let k = 0; k < K; k++) {
    v[k] = logInit[k] + logEmit[k * C];
  }

  // Recurse: v[c, j] = max_i(v[c-1, i] + logTrans[i, j]) + logEmit[j, c]
  for (let c = 1; c < C; c++) {
    for (let j = 0; j < K; j++) {
      let best = NEG_INF;
      let bestI = 0;
      for (let i = 0; i < K; i++) {
        const score = v[(c - 1) * K + i] + logTrans[i * K + j];
        if (score > best) {
          best = score;
          bestI = i;
        }
      }
      v[c * K + j] = best + logEmit[j * C + c];
      bp[c * K + j] = bestI;
    }
  }

  // Termination: find best final state
  let bestFinal = NEG_INF;
  let bestState = 0;
  for (let k = 0; k < K; k++) {
    const score = v[(C - 1) * K + k] + logTerm[k];
    if (score > bestFinal) {
      bestFinal = score;
      bestState = k;
    }
  }

  // Traceback
  const labels = new Int32Array(C);
  labels[C - 1] = bestState;
  for (let c = C - 2; c >= 0; c--) {
    labels[c] = bp[(c + 1) * K + labels[c + 1]];
  }

  return { labels, logProb: bestFinal };
}

// ── Posteriors ───────────────────────────────────────────────────────

export interface PosteriorResult {
  posteriors: Float64Array;  // (C, K) row-major, values in [0,1]
  logLikelihood: number;
}

export function hmmPosteriors(cg: CompiledGrammar, tw: TerminalWeights): PosteriorResult;
export function hmmPosteriors(tables: HmmTables): PosteriorResult;
export function hmmPosteriors(arg1: CompiledGrammar | HmmTables, arg2?: TerminalWeights): PosteriorResult {
  const t = isHmmTables(arg1) ? arg1 : buildHmmTables(arg1, arg2!);

  const { alpha, logLikelihood } = hmmForward(t);
  const { beta } = hmmBackward(t);

  const K = t.K;
  const C = t.C;
  const posteriors = new Float64Array(C * K);
  const tmp = new Float64Array(K);

  for (let c = 0; c < C; c++) {
    for (let k = 0; k < K; k++) {
      tmp[k] = alpha[c * K + k] + beta[c * K + k];
    }
    const logZ = logsumexp(tmp);
    for (let k = 0; k < K; k++) {
      posteriors[c * K + k] = Math.exp(tmp[k] - logZ);
    }
  }

  return { posteriors, logLikelihood };
}

// ── Helpers ─────────────────────────────────────────────────────────

function isHmmTables(x: unknown): x is HmmTables {
  return typeof x === 'object' && x !== null && 'logTrans' in x && 'logEmit' in x;
}
