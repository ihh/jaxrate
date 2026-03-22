/**
 * Pre-built grammars: pfold, gene_finder, pseudoknot.
 * Mirrors jaxrate/presets.py.
 */

import type { Grammar, EmissionGroup } from './types.js';
import { GrammarBuilder } from './grammar.js';

const emit1 = (model: number): EmissionGroup => ({ nPositions: 1, modelIndex: model });
const emit2 = (model: number): EmissionGroup => ({ nPositions: 2, modelIndex: model });

/**
 * RNA structure prediction SCFG (pfold-like).
 * Models: 0=unpaired, 1=paired.
 *
 * S → L S | F
 * L → e₁ S e₂ (paired)
 * F → e F | e (unpaired)
 */
export function pfoldGrammar(): Grammar {
  const gb = new GrammarBuilder();
  const S = gb.addNonterminal('S');
  const L = gb.addNonterminal('L');
  const F = gb.addNonterminal('F');
  const Fstar = gb.addNonterminal('F*');

  // S → L S (bifurcation into stem + continue)
  gb.addRule(S, [L, S], [], Math.log(0.4));
  // S → F (single-stranded)
  gb.addRule(S, [F], [], Math.log(0.5));
  // S → ε
  gb.addRule(S, [], [], Math.log(0.1));

  // L → e₁ Fstar e₂ (paired emission)
  gb.addRule(L, [Fstar], [emit2(1)], 0.0);

  // F* → L S (continue stem or start new)
  gb.addRule(Fstar, [L, S], [], Math.log(0.3));
  // F* → F (switch to unpaired)
  gb.addRule(Fstar, [F], [], Math.log(0.6));
  // F* → ε
  gb.addRule(Fstar, [], [], Math.log(0.1));

  // F → e F (left-emit unpaired)
  gb.addRule(F, [F], [emit1(0)], Math.log(0.8));
  // F → e (terminal unpaired)
  gb.addRule(F, [], [emit1(0)], Math.log(0.2));

  return gb.build(S, 2);
}

/**
 * Gene finder HMM.
 * Models: 0=intergenic, 1=exon, 2=intron.
 */
export function geneFinderGrammar(): Grammar {
  const gb = new GrammarBuilder();
  const IG = gb.addNonterminal('intergenic');
  const E1 = gb.addNonterminal('exon_1');
  const E2 = gb.addNonterminal('exon_2');
  const E3 = gb.addNonterminal('exon_3');
  const I = gb.addNonterminal('intron');

  // Intergenic
  gb.addRule(IG, [IG], [emit1(0)], Math.log(0.95));
  gb.addRule(IG, [E1], [emit1(0)], Math.log(0.04));
  gb.addRule(IG, [], [emit1(0)], Math.log(0.01));

  // Exon phase 1→2→3→1 (codon cycle)
  gb.addRule(E1, [E2], [emit1(1)], Math.log(0.85));
  gb.addRule(E1, [], [emit1(1)], Math.log(0.15));

  gb.addRule(E2, [E3], [emit1(1)], Math.log(0.98));
  gb.addRule(E2, [], [emit1(1)], Math.log(0.02));

  gb.addRule(E3, [E1], [emit1(1)], Math.log(0.80));
  gb.addRule(E3, [I], [emit1(1)], Math.log(0.10));
  gb.addRule(E3, [IG], [emit1(1)], Math.log(0.08));
  gb.addRule(E3, [], [emit1(1)], Math.log(0.02));

  // Intron
  gb.addRule(I, [I], [emit1(2)], Math.log(0.95));
  gb.addRule(I, [E1], [emit1(2)], Math.log(0.04));
  gb.addRule(I, [], [emit1(2)], Math.log(0.01));

  return gb.build(IG, 3);
}

/**
 * Pseudoknot MCFG (fan-out 2).
 * Models: 0=unpaired, 1=paired.
 */
export function pseudoknotGrammar(maxSpan = -1): Grammar {
  const gb = new GrammarBuilder();
  const S = gb.addNonterminal('S');
  const L = gb.addNonterminal('L');
  const PK = gb.addNonterminal('PK', 2, maxSpan);
  const PKinner = gb.addNonterminal('PK*', 2, maxSpan);

  // S → L S
  gb.addRule(S, [L, S], [], Math.log(0.3));
  // S → e S (single-stranded)
  gb.addRule(S, [S], [emit1(0)], Math.log(0.3));
  // S → PK (start pseudoknot — concat fan-out 2 → fan-out 1)
  gb.addRule(S, [PK], [], Math.log(0.1));
  // S → e
  gb.addRule(S, [], [emit1(0)], Math.log(0.2));
  // S → ε
  gb.addRule(S, [], [], Math.log(0.1));

  // L → e₁ S e₂ (paired)
  gb.addRule(L, [S], [emit2(1)], 0.0);

  // PK → PK* (unary)
  gb.addRule(PK, [PKinner], [], 0.0);

  // PK*(ax, by) → PK*(x, y) with composition 'll'
  gb.addRule(PKinner, [PKinner], [emit2(1)], Math.log(0.6), 1);
  // PK*(xa, yb) → PK*(x, y) with composition 'rr'
  gb.addRule(PKinner, [PKinner], [emit2(1)], Math.log(0.2), 2);
  // PK*(x, y) → S(x) S(y) (base case: two fan-out 1 regions)
  gb.addRule(PKinner, [S, S], [], Math.log(0.2));

  return gb.build(S, 2);
}
