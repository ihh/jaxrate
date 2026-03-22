/**
 * SCFG algorithms: inside, outside, CYK/Viterbi, posteriors.
 * O(C³K³) chart-based — mirrors jaxrate/scfg.py.
 */

import type { CompiledGrammar, TerminalWeights, ParseResult } from './types.js';
import { NEG_INF, logaddexp, logsumexp, negInfArray } from './log-semiring.js';
import { classifyRules, type ClassifiedRules } from './grammar.js';

// ── Chart indexing helpers ──────────────────────────────────────────

/** Chart shape: (K, C+1, C+1). chart[A * S*S + i * S + j] where S = C+1. */
function chartIdx(A: number, i: number, j: number, S: number): number {
  return A * S * S + i * S + j;
}

// ── Inside algorithm ────────────────────────────────────────────────

export interface InsideResult {
  alpha: Float64Array;  // (K, C+1, C+1) flattened
  logLikelihood: number;
  S: number;  // C+1, for indexing
}

export function scfgInside(cg: CompiledGrammar, tw: TerminalWeights): InsideResult {
  const K = cg.K;
  const C = tw.C;
  const S = C + 1;
  const alpha = negInfArray(K * S * S);
  const rules = classifyRules(cg);

  // Span 0: epsilon rules
  for (const r of rules.epsilon) {
    const lhs = cg.ruleLhs[r];
    const logW = cg.ruleLogWeights[r];
    for (let i = 0; i <= C; i++) {
      const idx = chartIdx(lhs, i, i, S);
      alpha[idx] = logaddexp(alpha[idx], logW);
    }
  }
  // Propagate unary at span 0
  propagateUnary(alpha, rules.unary, cg, S, 0, C);

  // Fill spans 1..C
  for (let span = 1; span <= C; span++) {
    for (let i = 0; i <= C - span; i++) {
      const j = i + span;

      // Terminal rules (span 1 only)
      if (span === 1) {
        for (const r of rules.terminal) {
          const lhs = cg.ruleLhs[r];
          const logW = cg.ruleLogWeights[r];
          const model = cg.ruleEmissionModel[r][0];
          const idx = chartIdx(lhs, i, j, S);
          alpha[idx] = logaddexp(alpha[idx], logW + tw.single[model][i]);
        }
      }

      // Left-emit unary: A → e B
      for (const r of rules.emitUnary) {
        const lhs = cg.ruleLhs[r];
        const rhs = cg.ruleRhs[r][0];
        const logW = cg.ruleLogWeights[r];
        const model = cg.ruleEmissionModel[r][0];
        const emitW = tw.single[model][i];
        const childVal = alpha[chartIdx(rhs, i + 1, j, S)];
        if (childVal > NEG_INF + 1e30) {
          const idx = chartIdx(lhs, i, j, S);
          alpha[idx] = logaddexp(alpha[idx], logW + emitW + childVal);
        }
      }

      // Paired emission: A → e₁ B e₂ (requires span >= 2)
      if (span >= 2 && tw.paired) {
        for (const r of rules.emitPaired) {
          const lhs = cg.ruleLhs[r];
          const rhs = cg.ruleRhs[r][0];
          const logW = cg.ruleLogWeights[r];
          const model = cg.ruleEmissionModel[r][0];
          const pairW = tw.paired[model][i * C + (j - 1)];
          const childVal = alpha[chartIdx(rhs, i + 1, j - 1, S)];
          if (childVal > NEG_INF + 1e30) {
            const idx = chartIdx(lhs, i, j, S);
            alpha[idx] = logaddexp(alpha[idx], logW + pairW + childVal);
          }
        }
      }

      // Binary: A → B C with split point k
      for (const r of rules.binary) {
        const lhs = cg.ruleLhs[r];
        const rhsB = cg.ruleRhs[r][0];
        const rhsC = cg.ruleRhs[r][1];
        const logW = cg.ruleLogWeights[r];
        const idx = chartIdx(lhs, i, j, S);

        for (let k = i; k <= j; k++) {
          const leftVal = alpha[chartIdx(rhsB, i, k, S)];
          const rightVal = alpha[chartIdx(rhsC, k, j, S)];
          if (leftVal > NEG_INF + 1e30 && rightVal > NEG_INF + 1e30) {
            alpha[idx] = logaddexp(alpha[idx], logW + leftVal + rightVal);
          }
        }
      }

      // Propagate unary at this span
      propagateUnaryAt(alpha, rules.unary, cg, S, i, j);
    }
  }

  const logLikelihood = alpha[chartIdx(cg.ruleLhs[0] === 0 ? 0 : 0, 0, C, S)];
  // Use grammar start symbol
  return { alpha, logLikelihood: alpha[chartIdx(0, 0, C, S)], S };
}

function propagateUnary(
  alpha: Float64Array, unaryRules: number[], cg: CompiledGrammar,
  S: number, spanStart: number, C: number
): void {
  for (let i = spanStart; i <= C; i++) {
    propagateUnaryAt(alpha, unaryRules, cg, S, i, i);
  }
}

function propagateUnaryAt(
  alpha: Float64Array, unaryRules: number[], cg: CompiledGrammar,
  S: number, i: number, j: number
): void {
  // Multiple passes to propagate through chains
  let changed = true;
  let passes = 0;
  while (changed && passes < cg.K) {
    changed = false;
    for (const r of unaryRules) {
      const lhs = cg.ruleLhs[r];
      const rhs = cg.ruleRhs[r][0];
      const logW = cg.ruleLogWeights[r];
      const childVal = alpha[chartIdx(rhs, i, j, S)];
      if (childVal > NEG_INF + 1e30) {
        const idx = chartIdx(lhs, i, j, S);
        const newVal = logaddexp(alpha[idx], logW + childVal);
        if (newVal > alpha[idx] + 1e-10) {
          alpha[idx] = newVal;
          changed = true;
        }
      }
    }
    passes++;
  }
}

// ── Outside algorithm ───────────────────────────────────────────────

export interface OutsideResult {
  beta: Float64Array;  // (K, C+1, C+1) flattened
  S: number;
}

export function scfgOutside(
  cg: CompiledGrammar, tw: TerminalWeights, insideResult: InsideResult
): OutsideResult {
  const K = cg.K;
  const C = tw.C;
  const S = C + 1;
  const { alpha } = insideResult;
  const beta = negInfArray(K * S * S);
  const rules = classifyRules(cg);

  // Initialize: beta[start, 0, C] = 0.0
  beta[chartIdx(0, 0, C, S)] = 0.0;

  // Top-down: decreasing span length
  for (let span = C; span >= 0; span--) {
    for (let i = 0; i <= C - span; i++) {
      const j = i + span;

      // Propagate unary outside (reverse direction)
      propagateUnaryOutside(beta, rules.unary, cg, S, i, j);

      // Binary rules: propagate outside to children
      for (const r of rules.binary) {
        const lhs = cg.ruleLhs[r];
        const rhsB = cg.ruleRhs[r][0];
        const rhsC = cg.ruleRhs[r][1];
        const logW = cg.ruleLogWeights[r];

        // For each split point, propagate outside from parent to children
        // Parent spans [i, j) — we need to find all parent spans that contain [i, j)
        // Actually, we iterate parent spans and propagate DOWN

        // Left child [i, k): parent [i, j) with right sibling [k, j)
        for (let k = i; k <= j; k++) {
          const parentBeta = beta[chartIdx(lhs, i, j, S)];
          if (parentBeta > NEG_INF + 1e30) {
            // Left child B spans [i, k)
            const rightInside = alpha[chartIdx(rhsC, k, j, S)];
            if (rightInside > NEG_INF + 1e30) {
              const bIdx = chartIdx(rhsB, i, k, S);
              beta[bIdx] = logaddexp(beta[bIdx], parentBeta + logW + rightInside);
            }
            // Right child C spans [k, j)
            const leftInside = alpha[chartIdx(rhsB, i, k, S)];
            if (leftInside > NEG_INF + 1e30) {
              const cIdx = chartIdx(rhsC, k, j, S);
              beta[cIdx] = logaddexp(beta[cIdx], parentBeta + logW + leftInside);
            }
          }
        }
      }

      // Left-emit unary: A → e B, parent [i, j), child [i+1, j)
      if (i + 1 <= j) {
        for (const r of rules.emitUnary) {
          const lhs = cg.ruleLhs[r];
          const rhs = cg.ruleRhs[r][0];
          const logW = cg.ruleLogWeights[r];
          const model = cg.ruleEmissionModel[r][0];
          const parentBeta = beta[chartIdx(lhs, i, j, S)];
          if (parentBeta > NEG_INF + 1e30) {
            const emitW = tw.single[model][i];
            const childIdx = chartIdx(rhs, i + 1, j, S);
            beta[childIdx] = logaddexp(beta[childIdx], parentBeta + logW + emitW);
          }
        }
      }

      // Paired emission: A → e₁ B e₂, parent [i, j), child [i+1, j-1)
      if (span >= 2 && tw.paired) {
        for (const r of rules.emitPaired) {
          const lhs = cg.ruleLhs[r];
          const rhs = cg.ruleRhs[r][0];
          const logW = cg.ruleLogWeights[r];
          const model = cg.ruleEmissionModel[r][0];
          const parentBeta = beta[chartIdx(lhs, i, j, S)];
          if (parentBeta > NEG_INF + 1e30) {
            const pairW = tw.paired[model][i * C + (j - 1)];
            const childIdx = chartIdx(rhs, i + 1, j - 1, S);
            beta[childIdx] = logaddexp(beta[childIdx], parentBeta + logW + pairW);
          }
        }
      }
    }
  }

  return { beta, S };
}

function propagateUnaryOutside(
  beta: Float64Array, unaryRules: number[], cg: CompiledGrammar,
  S: number, i: number, j: number
): void {
  // Reverse: from parent A → B, propagate beta[A] to beta[B]
  let changed = true;
  let passes = 0;
  while (changed && passes < cg.K) {
    changed = false;
    // Process in reverse order for top-down propagation
    for (let idx = unaryRules.length - 1; idx >= 0; idx--) {
      const r = unaryRules[idx];
      const lhs = cg.ruleLhs[r];
      const rhs = cg.ruleRhs[r][0];
      const logW = cg.ruleLogWeights[r];
      const parentBeta = beta[chartIdx(lhs, i, j, S)];
      if (parentBeta > NEG_INF + 1e30) {
        const childBetaIdx = chartIdx(rhs, i, j, S);
        const newVal = logaddexp(beta[childBetaIdx], parentBeta + logW);
        if (newVal > beta[childBetaIdx] + 1e-10) {
          beta[childBetaIdx] = newVal;
          changed = true;
        }
      }
    }
    passes++;
  }
}

// ── CYK/Viterbi ────────────────────────────────────────────────────

interface Backpointer {
  rule: number;
  split?: number;   // split point for binary rules
}

export function scfgViterbi(cg: CompiledGrammar, tw: TerminalWeights): ParseResult {
  const K = cg.K;
  const C = tw.C;
  const S = C + 1;
  const v = negInfArray(K * S * S);
  const bp = new Map<number, Backpointer>();  // chart index → backpointer
  const rules = classifyRules(cg);

  // Span 0: epsilon rules
  for (const r of rules.epsilon) {
    const lhs = cg.ruleLhs[r];
    const logW = cg.ruleLogWeights[r];
    for (let i = 0; i <= C; i++) {
      const idx = chartIdx(lhs, i, i, S);
      if (logW > v[idx]) {
        v[idx] = logW;
        bp.set(idx, { rule: r });
      }
    }
  }
  propagateUnaryViterbi(v, bp, rules.unary, cg, S, 0, C, true);

  // Fill spans 1..C
  for (let span = 1; span <= C; span++) {
    for (let i = 0; i <= C - span; i++) {
      const j = i + span;

      // Terminal rules (span 1)
      if (span === 1) {
        for (const r of rules.terminal) {
          const lhs = cg.ruleLhs[r];
          const logW = cg.ruleLogWeights[r];
          const model = cg.ruleEmissionModel[r][0];
          const score = logW + tw.single[model][i];
          const idx = chartIdx(lhs, i, j, S);
          if (score > v[idx]) {
            v[idx] = score;
            bp.set(idx, { rule: r });
          }
        }
      }

      // Left-emit unary
      for (const r of rules.emitUnary) {
        const lhs = cg.ruleLhs[r];
        const rhs = cg.ruleRhs[r][0];
        const logW = cg.ruleLogWeights[r];
        const model = cg.ruleEmissionModel[r][0];
        const childVal = v[chartIdx(rhs, i + 1, j, S)];
        if (childVal > NEG_INF + 1e30) {
          const score = logW + tw.single[model][i] + childVal;
          const idx = chartIdx(lhs, i, j, S);
          if (score > v[idx]) {
            v[idx] = score;
            bp.set(idx, { rule: r });
          }
        }
      }

      // Paired emission
      if (span >= 2 && tw.paired) {
        for (const r of rules.emitPaired) {
          const lhs = cg.ruleLhs[r];
          const rhs = cg.ruleRhs[r][0];
          const logW = cg.ruleLogWeights[r];
          const model = cg.ruleEmissionModel[r][0];
          const pairW = tw.paired[model][i * C + (j - 1)];
          const childVal = v[chartIdx(rhs, i + 1, j - 1, S)];
          if (childVal > NEG_INF + 1e30) {
            const score = logW + pairW + childVal;
            const idx = chartIdx(lhs, i, j, S);
            if (score > v[idx]) {
              v[idx] = score;
              bp.set(idx, { rule: r });
            }
          }
        }
      }

      // Binary
      for (const r of rules.binary) {
        const lhs = cg.ruleLhs[r];
        const rhsB = cg.ruleRhs[r][0];
        const rhsC = cg.ruleRhs[r][1];
        const logW = cg.ruleLogWeights[r];
        const idx = chartIdx(lhs, i, j, S);

        for (let k = i; k <= j; k++) {
          const leftVal = v[chartIdx(rhsB, i, k, S)];
          const rightVal = v[chartIdx(rhsC, k, j, S)];
          if (leftVal > NEG_INF + 1e30 && rightVal > NEG_INF + 1e30) {
            const score = logW + leftVal + rightVal;
            if (score > v[idx]) {
              v[idx] = score;
              bp.set(idx, { rule: r, split: k });
            }
          }
        }
      }

      // Unary propagation
      propagateUnaryViterbiAt(v, bp, rules.unary, cg, S, i, j);
    }
  }

  const logProb = v[chartIdx(0, 0, C, S)];

  // Traceback
  const labels = new Int32Array(C);
  if (logProb > NEG_INF + 1e30) {
    traceback(labels, v, bp, cg, S, 0, 0, C);
  }

  return { labels, logProb };
}

function traceback(
  labels: Int32Array, v: Float64Array, bp: Map<number, Backpointer>,
  cg: CompiledGrammar, S: number,
  nt: number, i: number, j: number
): void {
  if (i >= j) return;

  const idx = chartIdx(nt, i, j, S);
  const b = bp.get(idx);
  if (!b) return;

  const r = b.rule;
  const nRhs = cg.ruleNRhs[r];
  const nEmit = cg.ruleNEmissions[r];
  const lhs = cg.ruleLhs[r];

  if (nRhs === 0 && nEmit > 0) {
    // Terminal: label this column
    labels[i] = lhs;
  } else if (nRhs === 1 && nEmit === 0) {
    // Unary: recurse into child
    traceback(labels, v, bp, cg, S, cg.ruleRhs[r][0], i, j);
  } else if (nRhs === 1 && nEmit > 0) {
    const nPos = cg.ruleEmissionNPos[r][0];
    if (nPos === 2) {
      // Paired: label left and right columns
      labels[i] = lhs;
      labels[j - 1] = lhs;
      traceback(labels, v, bp, cg, S, cg.ruleRhs[r][0], i + 1, j - 1);
    } else {
      // Left-emit
      labels[i] = lhs;
      traceback(labels, v, bp, cg, S, cg.ruleRhs[r][0], i + 1, j);
    }
  } else if (nRhs >= 2 && b.split !== undefined) {
    // Binary: recurse into both children
    traceback(labels, v, bp, cg, S, cg.ruleRhs[r][0], i, b.split);
    traceback(labels, v, bp, cg, S, cg.ruleRhs[r][1], b.split, j);
  }
}

function propagateUnaryViterbi(
  v: Float64Array, bp: Map<number, Backpointer>,
  unaryRules: number[], cg: CompiledGrammar,
  S: number, spanStart: number, C: number, isSpanZero: boolean
): void {
  if (isSpanZero) {
    for (let i = spanStart; i <= C; i++) {
      propagateUnaryViterbiAt(v, bp, unaryRules, cg, S, i, i);
    }
  }
}

function propagateUnaryViterbiAt(
  v: Float64Array, bp: Map<number, Backpointer>,
  unaryRules: number[], cg: CompiledGrammar,
  S: number, i: number, j: number
): void {
  let changed = true;
  let passes = 0;
  while (changed && passes < cg.K) {
    changed = false;
    for (const r of unaryRules) {
      const lhs = cg.ruleLhs[r];
      const rhs = cg.ruleRhs[r][0];
      const logW = cg.ruleLogWeights[r];
      const childVal = v[chartIdx(rhs, i, j, S)];
      if (childVal > NEG_INF + 1e30) {
        const score = logW + childVal;
        const idx = chartIdx(lhs, i, j, S);
        if (score > v[idx] + 1e-10) {
          v[idx] = score;
          bp.set(idx, { rule: r });
          changed = true;
        }
      }
    }
    passes++;
  }
}

// ── Posteriors ───────────────────────────────────────────────────────

export interface ScfgPosteriorResult {
  posteriors: Float64Array;  // (C, K) row-major, values in [0,1]
  logLikelihood: number;
}

export function scfgPosteriors(cg: CompiledGrammar, tw: TerminalWeights): ScfgPosteriorResult {
  const K = cg.K;
  const C = tw.C;
  const insideResult = scfgInside(cg, tw);
  const { alpha, logLikelihood, S } = insideResult;
  const { beta } = scfgOutside(cg, tw, insideResult);
  const rules = classifyRules(cg);

  const posteriors = new Float64Array(C * K);

  // Accumulate emission posteriors per column per nonterminal
  for (let i = 0; i < C; i++) {
    for (let j = i + 1; j <= C; j++) {
      // Terminal rules (span 1 only)
      if (j === i + 1) {
        for (const r of rules.terminal) {
          const lhs = cg.ruleLhs[r];
          const logW = cg.ruleLogWeights[r];
          const model = cg.ruleEmissionModel[r][0];
          const contribution = beta[chartIdx(lhs, i, j, S)] + logW + tw.single[model][i];
          if (contribution > NEG_INF + 1e30) {
            posteriors[i * K + lhs] = logaddexp(
              posteriors[i * K + lhs] || NEG_INF,
              contribution - logLikelihood
            );
          }
        }
      }

      // Left-emit unary: column i is emitted
      for (const r of rules.emitUnary) {
        const lhs = cg.ruleLhs[r];
        const rhs = cg.ruleRhs[r][0];
        const logW = cg.ruleLogWeights[r];
        const model = cg.ruleEmissionModel[r][0];
        const contribution = beta[chartIdx(lhs, i, j, S)] + logW +
          tw.single[model][i] + alpha[chartIdx(rhs, i + 1, j, S)];
        if (contribution > NEG_INF + 1e30) {
          const logPost = contribution - logLikelihood;
          posteriors[i * K + lhs] = logaddexp(
            posteriors[i * K + lhs] || NEG_INF, logPost
          );
        }
      }

      // Paired emission: columns i and j-1 are emitted
      if (j >= i + 2 && tw.paired) {
        for (const r of rules.emitPaired) {
          const lhs = cg.ruleLhs[r];
          const rhs = cg.ruleRhs[r][0];
          const logW = cg.ruleLogWeights[r];
          const model = cg.ruleEmissionModel[r][0];
          const contribution = beta[chartIdx(lhs, i, j, S)] + logW +
            tw.paired[model][i * C + (j - 1)] +
            alpha[chartIdx(rhs, i + 1, j - 1, S)];
          if (contribution > NEG_INF + 1e30) {
            const logPost = contribution - logLikelihood;
            posteriors[i * K + lhs] = logaddexp(
              posteriors[i * K + lhs] || NEG_INF, logPost
            );
            posteriors[(j - 1) * K + lhs] = logaddexp(
              posteriors[(j - 1) * K + lhs] || NEG_INF, logPost
            );
          }
        }
      }
    }
  }

  // Convert from log-posteriors to posteriors
  for (let c = 0; c < C; c++) {
    for (let k = 0; k < K; k++) {
      const val = posteriors[c * K + k];
      posteriors[c * K + k] = val > NEG_INF + 1e30 ? Math.exp(val) : 0;
    }
  }

  return { posteriors, logLikelihood };
}
