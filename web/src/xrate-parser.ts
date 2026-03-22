/**
 * Parser for xrate .eg grammar files (S-expression format).
 * Mirrors jaxrate/xrate_parser.py.
 */

import type { Grammar, EmissionGroup, Rule } from './types.js';
import { GrammarBuilder } from './grammar.js';
import { NEG_INF } from './log-semiring.js';

// ── S-expression types ──────────────────────────────────────────────

export type SExpr = string | SExpr[];

// ── Tokenizer & Parser ──────────────────────────────────────────────

function tokenize(text: string): string[] {
  // Strip comments (;; to end of line)
  text = text.replace(/;;[^\n]*/g, '');
  text = text.replace(/;[^\n]*/g, '');

  const tokens: string[] = [];
  let i = 0;
  while (i < text.length) {
    const ch = text[i];
    if (/\s/.test(ch)) { i++; continue; }
    if (ch === '(' || ch === ')') {
      tokens.push(ch);
      i++;
    } else {
      let j = i;
      while (j < text.length && !/[\s()]/.test(text[j])) j++;
      tokens.push(text.substring(i, j));
      i = j;
    }
  }
  return tokens;
}

function parseTokens(tokens: string[], pos: number): [SExpr, number] {
  if (pos >= tokens.length) throw new Error('Unexpected end of input');

  if (tokens[pos] === '(') {
    const list: SExpr[] = [];
    pos++; // skip (
    while (pos < tokens.length && tokens[pos] !== ')') {
      const [expr, newPos] = parseTokens(tokens, pos);
      list.push(expr);
      pos = newPos;
    }
    if (pos >= tokens.length) throw new Error('Missing closing )');
    pos++; // skip )
    return [list, pos];
  } else if (tokens[pos] === ')') {
    throw new Error('Unexpected )');
  } else {
    return [tokens[pos], pos + 1];
  }
}

export function parseSExpr(text: string): SExpr[] {
  const tokens = tokenize(text);
  const exprs: SExpr[] = [];
  let pos = 0;
  while (pos < tokens.length) {
    const [expr, newPos] = parseTokens(tokens, pos);
    exprs.push(expr);
    pos = newPos;
  }
  return exprs;
}

// ── S-expression query helpers ──────────────────────────────────────

function findChild(expr: SExpr[], tag: string): SExpr[] | null {
  for (const child of expr) {
    if (Array.isArray(child) && child.length > 0 && child[0] === tag) {
      return child;
    }
  }
  return null;
}

function findAllChildren(expr: SExpr[], tag: string): SExpr[][] {
  return expr.filter(
    (c): c is SExpr[] => Array.isArray(c) && c.length > 0 && c[0] === tag
  );
}

function getAtom(expr: SExpr): string {
  if (typeof expr === 'string') return expr;
  throw new Error(`Expected atom, got list: ${JSON.stringify(expr)}`);
}

function getList(expr: SExpr): SExpr[] {
  if (Array.isArray(expr)) return expr;
  throw new Error(`Expected list, got atom: ${expr}`);
}

// ── Chain (substitution model) parsing ──────────────────────────────

export interface ChainModel {
  terminals: string[];    // e.g. ['NUC'] or ['LNUC', 'RNUC']
  alphabet: string[];     // e.g. ['a', 'c', 'g', 'u']
  pi: Float64Array;       // initial distribution
  rateMatrix: Float64Array; // Q matrix, row-major
  stateSize: number;      // alphabet.length ^ terminals.length
}

function parseChain(chainExpr: SExpr[], alphabet: string[]): ChainModel {
  const terminalExpr = findChild(chainExpr, 'terminal');
  if (!terminalExpr) throw new Error('Chain missing terminal declaration');

  const terminals = getList(terminalExpr[1]).map(getAtom);
  const nPos = terminals.length;
  const A = alphabet.length;
  const stateSize = Math.pow(A, nPos);

  const pi = new Float64Array(stateSize);
  const rateMatrix = new Float64Array(stateSize * stateSize);

  // Parse initial distribution
  for (const init of findAllChildren(chainExpr, 'initial')) {
    const stateExpr = findChild(init as SExpr[], 'state');
    const probExpr = findChild(init as SExpr[], 'prob');
    if (!stateExpr || !probExpr) continue;

    const stateTokens = getList(stateExpr[1]).map(getAtom);
    const stateIdx = stateToIndex(stateTokens, alphabet, nPos);
    pi[stateIdx] = parseFloat(getAtom(probExpr[1]));
  }

  // Parse mutation rates
  for (const mut of findAllChildren(chainExpr, 'mutate')) {
    const fromExpr = findChild(mut as SExpr[], 'from');
    const toExpr = findChild(mut as SExpr[], 'to');
    const rateExpr = findChild(mut as SExpr[], 'rate');
    if (!fromExpr || !toExpr || !rateExpr) continue;

    const fromTokens = getList(fromExpr[1]).map(getAtom);
    const toTokens = getList(toExpr[1]).map(getAtom);
    const fromIdx = stateToIndex(fromTokens, alphabet, nPos);
    const toIdx = stateToIndex(toTokens, alphabet, nPos);
    const rate = parseFloat(getAtom(rateExpr[1]));
    rateMatrix[fromIdx * stateSize + toIdx] = rate;
  }

  // Fill diagonal: Q[i,i] = -sum_{j≠i} Q[i,j]
  for (let i = 0; i < stateSize; i++) {
    let rowSum = 0;
    for (let j = 0; j < stateSize; j++) {
      if (j !== i) rowSum += rateMatrix[i * stateSize + j];
    }
    rateMatrix[i * stateSize + i] = -rowSum;
  }

  return { terminals, alphabet, pi, rateMatrix, stateSize };
}

function stateToIndex(tokens: string[], alphabet: string[], nPos: number): number {
  let idx = 0;
  for (let p = 0; p < nPos; p++) {
    const a = alphabet.indexOf(tokens[p].toLowerCase());
    if (a < 0) throw new Error(`Unknown token: ${tokens[p]}`);
    idx = idx * alphabet.length + a;
  }
  return idx;
}

// ── Transform (production rule) parsing ─────────────────────────────

interface ParsedTransform {
  from: string;
  to: string[];
  prob: number;
}

function parseTransform(
  transformExpr: SExpr[],
  chainTerminalMap: Map<string, { modelIndex: number; nPositions: number }>
): { rule: Omit<Rule, 'lhs'> & { lhsName: string; rhsNames: string[] } } {
  const fromExpr = findChild(transformExpr, 'from');
  const toExpr = findChild(transformExpr, 'to');
  const probExpr = findChild(transformExpr, 'prob');

  if (!fromExpr) throw new Error('Transform missing from');

  const lhsName = getAtom(getList(fromExpr[1])[0]);
  const toSymbols: string[] = toExpr
    ? getList(toExpr[1]).map(getAtom)
    : [];
  const prob = probExpr ? parseFloat(getAtom(probExpr[1])) : 1.0;
  const logWeight = prob > 0 ? Math.log(prob) : NEG_INF;

  // Classify symbols as terminals or nonterminals
  const rhsNames: string[] = [];
  const emissions: EmissionGroup[] = [];

  for (const sym of toSymbols) {
    const chainInfo = chainTerminalMap.get(sym);
    if (chainInfo) {
      // This is a terminal (chain pseudoterminal)
      emissions.push({ nPositions: chainInfo.nPositions, modelIndex: chainInfo.modelIndex });
    } else {
      // This is a nonterminal
      rhsNames.push(sym);
    }
  }

  return {
    rule: {
      lhsName,
      rhsNames,
      rhs: [],  // filled later
      emissions,
      logWeight,
      composition: 0,
    },
  };
}

// ── Main parser ─────────────────────────────────────────────────────

export interface XrateGrammar {
  grammar: Grammar;
  chains: ChainModel[];
  alphabet: string[];
  nonterminalNames: string[];
}

export function parseXrate(text: string, startNonterminal?: string): XrateGrammar {
  const exprs = parseSExpr(text);

  // Find alphabet
  let alphabet: string[] = ['a', 'c', 'g', 'u'];  // default RNA
  for (const expr of exprs) {
    if (!Array.isArray(expr)) continue;
    if (expr[0] === 'alphabet') {
      const tokenExpr = findChild(expr, 'token');
      if (tokenExpr) {
        alphabet = getList(tokenExpr[1]).map(getAtom);
      }
    }
  }

  // Find grammar block
  let grammarBlock: SExpr[] | null = null;
  for (const expr of exprs) {
    if (!Array.isArray(expr)) continue;
    if (expr[0] === 'grammar') {
      grammarBlock = expr;
      break;
    }
  }
  if (!grammarBlock) throw new Error('No grammar block found');

  // Parse chains
  const chains: ChainModel[] = [];
  const chainTerminalMap = new Map<string, { modelIndex: number; nPositions: number }>();
  let singleModelIdx = 0;
  let pairedModelIdx = 0;

  for (const chainExpr of findAllChildren(grammarBlock, 'chain')) {
    const chain = parseChain(chainExpr, alphabet);
    chains.push(chain);

    const nPos = chain.terminals.length;
    const modelIdx = nPos === 1 ? singleModelIdx++ : pairedModelIdx++;

    for (const term of chain.terminals) {
      chainTerminalMap.set(term, { modelIndex: modelIdx, nPositions: nPos });
    }
  }

  // Parse transforms
  const transforms = findAllChildren(grammarBlock, 'transform');
  const parsedRules: Array<ReturnType<typeof parseTransform>['rule']> = [];

  for (const tExpr of transforms) {
    const { rule } = parseTransform(tExpr, chainTerminalMap);
    parsedRules.push(rule);
  }

  // Collect nonterminal names
  const ntNameSet = new Set<string>();
  for (const r of parsedRules) {
    ntNameSet.add(r.lhsName);
    for (const name of r.rhsNames) ntNameSet.add(name);
  }

  // Build grammar
  const gb = new GrammarBuilder();
  const ntMap = new Map<string, number>();
  const ntNames: string[] = [];

  // Add start NT first if specified
  if (startNonterminal && ntNameSet.has(startNonterminal)) {
    const idx = gb.addNonterminal(startNonterminal);
    ntMap.set(startNonterminal, idx);
    ntNames.push(startNonterminal);
  }

  // Add remaining NTs
  for (const name of ntNameSet) {
    if (!ntMap.has(name)) {
      const idx = gb.addNonterminal(name);
      ntMap.set(name, idx);
      ntNames.push(name);
    }
  }

  // Build rules
  const nModels = Math.max(singleModelIdx, pairedModelIdx, 1);
  for (const r of parsedRules) {
    const lhs = ntMap.get(r.lhsName)!;
    const rhs = r.rhsNames.map(n => ntMap.get(n)!);
    gb.addRule(lhs, rhs, r.emissions, r.logWeight, r.composition);
  }

  const start = startNonterminal ? ntMap.get(startNonterminal) ?? 0 : 0;
  const grammar = gb.build(start, nModels);

  return { grammar, chains, alphabet, nonterminalNames: ntNames };
}

export function parseXrateFile(text: string): XrateGrammar {
  return parseXrate(text);
}
