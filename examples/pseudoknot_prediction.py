"""Pseudoknot prediction using MCFG with max_span constraint.

Demonstrates:
1. Loading an Rfam pseudoknot alignment (RF00390 TYMV_upPK)
2. Parsing SS_cons annotation to extract crossing base pairs
3. Creating synthetic terminal weights from known structure
4. Using pseudoknot_grammar(max_span=...) to predict structure
5. Showing max_span's effect on log-likelihood
"""

import os
import jax.numpy as jnp
import numpy as np

from jaxrate import compile_grammar, TerminalWeights, viterbi
from jaxrate.presets import pseudoknot_grammar
from jaxrate.mcfg import mcfg_inside, mcfg_viterbi


def parse_stockholm(path):
    """Parse a Stockholm alignment file.

    Returns:
        seqs: dict of name -> sequence
        ss_cons: SS_cons string
    """
    seqs = {}
    ss_cons = None
    with open(path) as f:
        for line in f:
            line = line.rstrip()
            if line.startswith('#') or line.startswith('//') or not line.strip():
                if line.startswith('#=GC SS_cons'):
                    ss_cons = line.split()[-1]
                continue
            parts = line.split()
            if len(parts) == 2:
                seqs[parts[0]] = parts[1]
    return seqs, ss_cons


def parse_ss_cons(ss_cons):
    """Parse SS_cons pseudoknot annotation into base pairs.

    Handles uppercase/lowercase letter pairs (A/a, B/b) and bracket pairs (</>).

    Returns:
        pairs: sorted list of (i, j) base pairs (0-indexed)
    """
    pairs = []
    stack_bracket = []
    stacks = {}  # uppercase letter -> stack of positions

    for i, ch in enumerate(ss_cons):
        if ch in '<([{':
            stack_bracket.append(i)
        elif ch in '>)]}':
            if stack_bracket:
                j = stack_bracket.pop()
                pairs.append((j, i))
        elif ch.isupper():
            letter = ch.upper()
            if letter not in stacks:
                stacks[letter] = []
            stacks[letter].append(i)
        elif ch.islower():
            letter = ch.upper()
            if letter in stacks and stacks[letter]:
                j = stacks[letter].pop(0)
                pairs.append((j, i))

    return sorted(pairs)


def main():
    # --- Part 1: RF00390 (TYMV_upPK) pseudoknot ---
    data_dir = os.path.join(os.path.dirname(__file__), '..', 'jaxrate', 'data')
    sto_path = os.path.join(data_dir, 'RF00390.sto')

    seqs, ss_cons = parse_stockholm(sto_path)
    print("=== RF00390 (TYMV_upPK) H-type pseudoknot ===")
    print(f"Sequences: {len(seqs)}")
    print(f"SS_cons: {ss_cons}")

    pairs = parse_ss_cons(ss_cons)
    print(f"\nBase pairs ({len(pairs)}):")
    for i, j in pairs:
        print(f"  ({i:2d}, {j:2d})  {ss_cons[i]}-{ss_cons[j]}")

    # Verify crossing
    crossing = False
    for k, (i1, j1) in enumerate(pairs):
        for i2, j2 in pairs[k + 1:]:
            if i1 < i2 < j1 < j2:
                crossing = True
                break
    print(f"Crossing stems: {crossing}")

    # Use effective length (strip trailing dots)
    ss_trimmed = ss_cons.rstrip('.')
    C = len(ss_trimmed)
    print(f"Effective length: {C}")

    # Create terminal weights favoring known pairs
    single = np.full((2, C), -2.0)
    paired = np.full((2, C, C), -10.0)
    for i, j in pairs:
        if i < C and j < C:
            paired[1, i, j] = -0.1
    tw = TerminalWeights(single=jnp.array(single), paired=jnp.array(paired), C=C)

    # Predict structure
    grammar = pseudoknot_grammar(max_span=15)
    cg = compile_grammar(grammar)
    print(f"\nGrammar: {cg.grammar_class}, {cg.n_nonterminals} NTs, {cg.n_rules} rules")

    labels, log_prob, bp = mcfg_viterbi(cg, tw)
    nt_names = [nt.name for nt in grammar.nonterminals]
    # Single-char abbreviation for display: S='.', L='(', PK='X'
    abbrev = {'S': '.', 'L': '(', 'PK': 'X'}
    annotation = [nt_names[int(l)] for l in labels]
    ann_chars = [abbrev.get(a, a[0]) for a in annotation]

    seq = list(seqs.values())[0][:C]
    print(f"\nSequence:   {seq}")
    print(f"SS_cons:    {ss_trimmed}")
    print(f"Predicted:  {''.join(ann_chars)}   (.=S, (=L, X=PK)")
    print(f"Log-prob:   {log_prob:.4f}")

    # Accuracy
    paired_cols = set()
    for i, j in pairs:
        if i < C and j < C:
            paired_cols.update([i, j])
    predicted_paired = {i for i in range(C) if annotation[i] in ('L', 'PK')}
    true_paired = paired_cols & set(range(C))
    overlap = predicted_paired & true_paired
    sens = len(overlap) / len(true_paired) if true_paired else 0
    ppv = len(overlap) / len(predicted_paired) if predicted_paired else 0
    print(f"Paired: true={len(true_paired)}, pred={len(predicted_paired)}, "
          f"overlap={len(overlap)}")
    print(f"Sensitivity={sens:.2f}, PPV={ppv:.2f}")

    # --- Part 2: max_span effect on a targeted example ---
    print("\n=== max_span constraint demonstration ===")

    # Create a short sequence where PK's two stems are used
    # Layout: pair(0,2) pair(3,5) — two adjacent 3-nt stems
    C2 = 6
    single2 = np.full((2, C2), -2.0)
    paired2 = np.full((2, C2, C2), -10.0)
    paired2[1, 0, 2] = -0.1   # stem 1
    paired2[1, 3, 5] = -0.1   # stem 2
    tw2 = TerminalWeights(single=jnp.array(single2), paired=jnp.array(paired2), C=C2)

    print(f"\nSequence length: {C2}")
    print("Pairs: (0,2), (3,5)")
    for ms in [None, 5, 3, 2, 1]:
        g = pseudoknot_grammar(max_span=ms)
        cg_ms = compile_grammar(g)
        _, _, ll = mcfg_inside(cg_ms, tw2)
        label = f"max_span={ms}" if ms is not None else "unlimited"
        print(f"  {label:>15s}: LL = {float(ll):.4f}")


if __name__ == '__main__':
    main()
