/**
 * Grammar construction, validation, compilation, and classification.
 * Mirrors jaxrate/grammar.py.
 */

import type {
  Nonterminal, EmissionGroup, Rule, Grammar,
  CompiledGrammar, GrammarClass,
} from './types.js';

// ── GrammarBuilder ──────────────────────────────────────────────────

export class GrammarBuilder {
  private nonterminals: Nonterminal[] = [];
  private rules: Rule[] = [];

  addNonterminal(name: string, fanOut = 1, maxSpan = -1): number {
    const idx = this.nonterminals.length;
    this.nonterminals.push({ name, fanOut, maxSpan });
    return idx;
  }

  addRule(
    lhs: number,
    rhs: number[] = [],
    emissions: EmissionGroup[] = [],
    logWeight = 0.0,
    composition = 0,
  ): void {
    this.rules.push({ lhs, rhs, emissions, logWeight, composition });
  }

  build(start = 0, nModels = 1): Grammar {
    return {
      nonterminals: [...this.nonterminals],
      rules: [...this.rules],
      start,
      nModels,
    };
  }
}

// ── Classification ──────────────────────────────────────────────────

export function classifyGrammar(grammar: Grammar): GrammarClass {
  // MCFG if any nonterminal has fan_out > 1
  if (grammar.nonterminals.some(nt => nt.fanOut > 1)) return 'mcfg';

  // HMM if all rules are right-linear (at most 1 RHS NT, at right end)
  const isRightLinear = grammar.rules.every(rule => {
    if (rule.rhs.length > 1) return false;
    // If there's 1 RHS NT, emissions must precede it (left-emit)
    // This is always true by construction
    return true;
  });
  // Also check: no paired emissions, no bifurcation
  const hasPaired = grammar.rules.some(r =>
    r.emissions.some(e => e.nPositions === 2)
  );
  const hasBifurcation = grammar.rules.some(r => r.rhs.length >= 2);

  if (!hasPaired && !hasBifurcation && isRightLinear) return 'hmm';
  return 'scfg';
}

// ── Validation ──────────────────────────────────────────────────────

export function validateGrammar(grammar: Grammar): void {
  const K = grammar.nonterminals.length;
  if (K === 0) throw new Error('Grammar has no nonterminals');
  if (grammar.start < 0 || grammar.start >= K) {
    throw new Error(`Invalid start nonterminal: ${grammar.start}`);
  }
  if (grammar.nonterminals[grammar.start].fanOut !== 1) {
    throw new Error('Start nonterminal must have fan_out=1');
  }
  for (const rule of grammar.rules) {
    if (rule.lhs < 0 || rule.lhs >= K) {
      throw new Error(`Rule LHS ${rule.lhs} out of bounds`);
    }
    for (const r of rule.rhs) {
      if (r < 0 || r >= K) throw new Error(`Rule RHS ${r} out of bounds`);
    }
    for (const e of rule.emissions) {
      if (e.modelIndex < 0 || e.modelIndex >= grammar.nModels) {
        throw new Error(`Emission model ${e.modelIndex} out of bounds`);
      }
    }
  }
}

// ── Compilation ─────────────────────────────────────────────────────

export function compileGrammar(grammar: Grammar): CompiledGrammar {
  validateGrammar(grammar);

  const K = grammar.nonterminals.length;
  const nRules = grammar.rules.length;
  const grammarClass = classifyGrammar(grammar);

  const ruleLhs = new Int32Array(nRules);
  const ruleNRhs = new Int32Array(nRules);
  const ruleLogWeights = new Float64Array(nRules);
  const ruleNEmissions = new Int32Array(nRules);
  const ruleComposition = new Int32Array(nRules);
  const ruleRhs: Int32Array[] = [];
  const ruleEmissionModel: Int32Array[] = [];
  const ruleEmissionNPos: Int32Array[] = [];

  // Per-nonterminal rule lists
  const ntRules: number[][] = Array.from({ length: K }, () => []);

  for (let r = 0; r < nRules; r++) {
    const rule = grammar.rules[r];
    ruleLhs[r] = rule.lhs;
    ruleNRhs[r] = rule.rhs.length;
    ruleLogWeights[r] = rule.logWeight;
    ruleNEmissions[r] = rule.emissions.length;
    ruleComposition[r] = rule.composition;
    ruleRhs.push(new Int32Array(rule.rhs));
    ruleEmissionModel.push(new Int32Array(rule.emissions.map(e => e.modelIndex)));
    ruleEmissionNPos.push(new Int32Array(rule.emissions.map(e => e.nPositions)));
    ntRules[rule.lhs].push(r);
  }

  const rulesForNt = ntRules.map(arr => new Int32Array(arr));
  const fanOuts = new Int32Array(K);
  const maxSpans = new Int32Array(K);
  for (let k = 0; k < K; k++) {
    fanOuts[k] = grammar.nonterminals[k].fanOut;
    maxSpans[k] = grammar.nonterminals[k].maxSpan;
  }

  return {
    grammarClass, K, nRules,
    ruleLhs, ruleRhs, ruleNRhs, ruleLogWeights,
    ruleEmissionModel, ruleEmissionNPos, ruleNEmissions,
    ruleComposition, rulesForNt, fanOuts, maxSpans,
  };
}

// ── Topological sort of unary rules ─────────────────────────────────

export interface ClassifiedRules {
  terminal: number[];    // A → e (emit, no RHS)
  unary: number[];       // A → B (1 RHS, no emit) — topologically sorted
  emitUnary: number[];   // A → e B (1 RHS, single emit)
  binary: number[];      // A → B C (2 RHS)
  emitPaired: number[];  // A → e₁ B e₂ (1 RHS, paired emit)
  epsilon: number[];     // A → ε (no RHS, no emit)
}

export function classifyRules(cg: CompiledGrammar): ClassifiedRules {
  const result: ClassifiedRules = {
    terminal: [], unary: [], emitUnary: [],
    binary: [], emitPaired: [], epsilon: [],
  };

  for (let r = 0; r < cg.nRules; r++) {
    const nRhs = cg.ruleNRhs[r];
    const nEmit = cg.ruleNEmissions[r];

    if (nRhs === 0 && nEmit === 0) {
      result.epsilon.push(r);
    } else if (nRhs === 0 && nEmit > 0) {
      result.terminal.push(r);
    } else if (nRhs === 1 && nEmit === 0) {
      result.unary.push(r);
    } else if (nRhs === 1 && nEmit > 0) {
      // Check if paired
      if (cg.ruleEmissionNPos[r][0] === 2) {
        result.emitPaired.push(r);
      } else {
        result.emitUnary.push(r);
      }
    } else if (nRhs >= 2) {
      result.binary.push(r);
    }
  }

  // Topologically sort unary rules (Kahn's algorithm)
  result.unary = topoSortUnary(result.unary, cg);

  return result;
}

function topoSortUnary(unaryRules: number[], cg: CompiledGrammar): number[] {
  if (unaryRules.length === 0) return [];

  // Build dependency graph: rule r depends on rules whose LHS = r's RHS
  const K = cg.K;
  const rulesByLhs = new Map<number, number[]>();
  for (const r of unaryRules) {
    const lhs = cg.ruleLhs[r];
    if (!rulesByLhs.has(lhs)) rulesByLhs.set(lhs, []);
    rulesByLhs.get(lhs)!.push(r);
  }

  // in-degree: how many unary rules have RHS that matches this rule's LHS
  const inDegree = new Map<number, number>();
  for (const r of unaryRules) inDegree.set(r, 0);

  for (const r of unaryRules) {
    const rhsNt = cg.ruleRhs[r][0];
    const deps = rulesByLhs.get(rhsNt);
    if (deps) {
      for (const dep of deps) {
        inDegree.set(r, (inDegree.get(r) ?? 0) + 1);
      }
    }
  }

  const queue: number[] = [];
  for (const r of unaryRules) {
    if (inDegree.get(r) === 0) queue.push(r);
  }

  const sorted: number[] = [];
  while (queue.length > 0) {
    const r = queue.shift()!;
    sorted.push(r);
    const lhs = cg.ruleLhs[r];
    // Find rules that depend on this one (rules whose RHS = lhs)
    for (const s of unaryRules) {
      if (cg.ruleRhs[s][0] === lhs && s !== r) {
        const deg = (inDegree.get(s) ?? 1) - 1;
        inDegree.set(s, deg);
        if (deg === 0) queue.push(s);
      }
    }
  }

  // If cycle detected, append remaining in original order
  if (sorted.length < unaryRules.length) {
    const inSorted = new Set(sorted);
    for (const r of unaryRules) {
      if (!inSorted.has(r)) sorted.push(r);
    }
  }

  return sorted;
}
