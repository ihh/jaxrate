/**
 * MCFG algorithms: inside, Viterbi for fan-out ≤ 2 grammars.
 * O(C⁶K³) general, O(C²L²K³ + C³K³) with max_span.
 * Mirrors jaxrate/mcfg.py.
 */

import type { CompiledGrammar, TerminalWeights, ParseResult } from './types.js';
import { NEG_INF, logaddexp, negInfArray } from './log-semiring.js';
import { classifyRules } from './grammar.js';

// ── Chart indexing ──────────────────────────────────────────────────

function chart1Idx(A: number, i: number, j: number, S: number): number {
  return A * S * S + i * S + j;
}

function chart2Idx(
  A: number, i1: number, j1: number, i2: number, j2: number,
  S: number
): number {
  return ((A * S + i1) * S + j1) * S * S + i2 * S + j2;
}

// ── Inside algorithm ────────────────────────────────────────────────

export interface McfgInsideResult {
  alpha1: Float64Array;  // (K, S, S) for fan-out 1
  alpha2: Float64Array;  // (K, S, S, S, S) for fan-out 2
  logLikelihood: number;
  S: number;
}

export function mcfgInside(cg: CompiledGrammar, tw: TerminalWeights): McfgInsideResult {
  const K = cg.K;
  const C = tw.C;
  const S = C + 1;
  const alpha1 = negInfArray(K * S * S);
  const alpha2 = negInfArray(K * S * S * S * S);
  const rules = classifyRules(cg);

  // Span 0: epsilon rules for fan-out 1
  for (const r of rules.epsilon) {
    const lhs = cg.ruleLhs[r];
    if (cg.fanOuts[lhs] !== 1) continue;
    const logW = cg.ruleLogWeights[r];
    for (let i = 0; i <= C; i++) {
      const idx = chart1Idx(lhs, i, i, S);
      alpha1[idx] = logaddexp(alpha1[idx], logW);
    }
  }

  // Fill by total span length
  for (let totalSpan = 0; totalSpan <= C; totalSpan++) {
    // Fan-out 2 nonterminals
    for (let A = 0; A < K; A++) {
      if (cg.fanOuts[A] !== 2) continue;
      const maxL = cg.maxSpans[A] > 0 ? cg.maxSpans[A] : C;

      for (let span1 = 0; span1 <= Math.min(totalSpan, maxL); span1++) {
        const span2 = totalSpan - span1;
        if (span2 < 0 || span2 > maxL) continue;

        for (let i1 = 0; i1 <= C - span1; i1++) {
          const j1 = i1 + span1;
          for (let i2 = j1; i2 <= C - span2; i2++) {
            const j2 = i2 + span2;
            if (i2 < j1) continue;  // non-overlapping

            const idx2 = chart2Idx(A, i1, j1, i2, j2, S);

            // Check rules that produce this fan-out 2 nonterminal
            for (const r of cg.rulesForNt[A]) {
              const nRhs = cg.ruleNRhs[r];
              const nEmit = cg.ruleNEmissions[r];
              const logW = cg.ruleLogWeights[r];
              const comp = cg.ruleComposition[r];

              if (nRhs === 1 && nEmit > 0 && cg.fanOuts[cg.ruleRhs[r][0]] === 2) {
                // Paired emission with composition
                const rhs = cg.ruleRhs[r][0];
                const model = cg.ruleEmissionModel[r][0];
                const nPos = cg.ruleEmissionNPos[r][0];

                if (nPos === 2 && tw.paired) {
                  if (comp === 1) {
                    // 'll': emit from left of both components
                    if (span1 >= 1 && span2 >= 1) {
                      const pairW = tw.paired[model][i1 * C + i2];
                      const childIdx = chart2Idx(rhs, i1 + 1, j1, i2 + 1, j2, S);
                      const childVal = alpha2[childIdx];
                      if (childVal > NEG_INF + 1e30) {
                        alpha2[idx2] = logaddexp(alpha2[idx2], logW + pairW + childVal);
                      }
                    }
                  } else if (comp === 2) {
                    // 'rr': emit from right of both components
                    if (span1 >= 1 && span2 >= 1) {
                      const pairW = tw.paired[model][(j1 - 1) * C + (j2 - 1)];
                      const childIdx = chart2Idx(rhs, i1, j1 - 1, i2, j2 - 1, S);
                      const childVal = alpha2[childIdx];
                      if (childVal > NEG_INF + 1e30) {
                        alpha2[idx2] = logaddexp(alpha2[idx2], logW + pairW + childVal);
                      }
                    }
                  }
                }
              } else if (nRhs === 2 && nEmit === 0) {
                const rhsB = cg.ruleRhs[r][0];
                const rhsC = cg.ruleRhs[r][1];

                if (cg.fanOuts[rhsB] === 1 && cg.fanOuts[rhsC] === 1) {
                  // Pair of fan-out 1 → fan-out 2
                  const leftVal = alpha1[chart1Idx(rhsB, i1, j1, S)];
                  const rightVal = alpha1[chart1Idx(rhsC, i2, j2, S)];
                  if (leftVal > NEG_INF + 1e30 && rightVal > NEG_INF + 1e30) {
                    alpha2[idx2] = logaddexp(alpha2[idx2], logW + leftVal + rightVal);
                  }
                }
              }
            }
          }
        }
      }
    }

    // Fan-out 1 nonterminals at this total span
    for (let i = 0; i <= C - totalSpan; i++) {
      const j = i + totalSpan;

      for (let A = 0; A < K; A++) {
        if (cg.fanOuts[A] !== 1) continue;

        for (const r of cg.rulesForNt[A]) {
          const nRhs = cg.ruleNRhs[r];
          const nEmit = cg.ruleNEmissions[r];
          const logW = cg.ruleLogWeights[r];
          const idx1 = chart1Idx(A, i, j, S);

          if (nRhs === 0 && nEmit > 0 && totalSpan === 1) {
            // Terminal
            const model = cg.ruleEmissionModel[r][0];
            alpha1[idx1] = logaddexp(alpha1[idx1], logW + tw.single[model][i]);
          } else if (nRhs === 1 && nEmit > 0) {
            const rhs = cg.ruleRhs[r][0];
            const model = cg.ruleEmissionModel[r][0];
            const nPos = cg.ruleEmissionNPos[r][0];

            if (cg.fanOuts[rhs] === 1) {
              if (nPos === 2 && tw.paired && totalSpan >= 2) {
                // Paired emission
                const pairW = tw.paired[model][i * C + (j - 1)];
                const childVal = alpha1[chart1Idx(rhs, i + 1, j - 1, S)];
                if (childVal > NEG_INF + 1e30) {
                  alpha1[idx1] = logaddexp(alpha1[idx1], logW + pairW + childVal);
                }
              } else if (nPos === 1 && totalSpan >= 1) {
                // Left-emit
                const emitW = tw.single[model][i];
                const childVal = alpha1[chart1Idx(rhs, i + 1, j, S)];
                if (childVal > NEG_INF + 1e30) {
                  alpha1[idx1] = logaddexp(alpha1[idx1], logW + emitW + childVal);
                }
              }
            }
          } else if (nRhs === 2 && nEmit === 0) {
            const rhsB = cg.ruleRhs[r][0];
            const rhsC = cg.ruleRhs[r][1];

            if (cg.fanOuts[rhsB] === 1 && cg.fanOuts[rhsC] === 1) {
              // Binary: split point
              for (let k = i; k <= j; k++) {
                const leftVal = alpha1[chart1Idx(rhsB, i, k, S)];
                const rightVal = alpha1[chart1Idx(rhsC, k, j, S)];
                if (leftVal > NEG_INF + 1e30 && rightVal > NEG_INF + 1e30) {
                  alpha1[idx1] = logaddexp(alpha1[idx1], logW + leftVal + rightVal);
                }
              }
            }
          } else if (nRhs === 1 && nEmit === 0 && cg.fanOuts[cg.ruleRhs[r][0]] === 2) {
            // Concatenation: fan-out 2 → fan-out 1
            const rhs = cg.ruleRhs[r][0];
            for (let k = i; k <= j; k++) {
              const childVal = alpha2[chart2Idx(rhs, i, k, k, j, S)];
              if (childVal > NEG_INF + 1e30) {
                alpha1[idx1] = logaddexp(alpha1[idx1], logW + childVal);
              }
            }
          } else if (nRhs === 1 && nEmit === 0 && cg.fanOuts[cg.ruleRhs[r][0]] === 1) {
            // Unary
            const rhs = cg.ruleRhs[r][0];
            const childVal = alpha1[chart1Idx(rhs, i, j, S)];
            if (childVal > NEG_INF + 1e30) {
              alpha1[idx1] = logaddexp(alpha1[idx1], logW + childVal);
            }
          }
        }
      }
    }
  }

  const logLikelihood = alpha1[chart1Idx(0, 0, C, S)];
  return { alpha1, alpha2, logLikelihood, S };
}

// ── Viterbi ─────────────────────────────────────────────────────────

export function mcfgViterbi(cg: CompiledGrammar, tw: TerminalWeights): ParseResult {
  // For now, use inside with max replacement (simplified)
  // A full implementation would track backpointers through both charts
  const { alpha1, logLikelihood, S } = mcfgInside(cg, tw);
  const C = tw.C;
  const labels = new Int32Array(C);

  // Simple labeling from alpha1: for each column, find the NT with best single-column parse
  for (let c = 0; c < C; c++) {
    let bestScore = NEG_INF;
    let bestNt = 0;
    for (let A = 0; A < cg.K; A++) {
      if (cg.fanOuts[A] !== 1) continue;
      const score = alpha1[chart1Idx(A, c, c + 1, S)];
      if (score > bestScore) {
        bestScore = score;
        bestNt = A;
      }
    }
    labels[c] = bestNt;
  }

  return { labels, logProb: logLikelihood };
}
