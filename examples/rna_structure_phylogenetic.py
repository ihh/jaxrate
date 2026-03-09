"""RNA structure prediction with real phylogenetic terminal weights.

Demonstrates the full jaxrate + subby pipeline:
1. Parse pfold.eg to get grammar + substitution models
2. Load Rfam alignment (RF00390 TYMV_upPK)
3. Build a star tree from the alignment
4. Compute phylogenetic terminal weights via subby
5. Run inside/Viterbi algorithms
6. Compare predicted structure to SS_cons annotation
"""

import os
import numpy as np
import jax
import jax.numpy as jnp

# Enable float64 for numerical accuracy
jax.config.update("jax_enable_x64", True)

from subby.jax import LogLike
from subby.jax.types import Tree
from subby.formats import parse_stockholm, all_column_ktuples, kmer_tokenize

from jaxrate import (
    compile_grammar, viterbi, TerminalWeights,
    parse_xrate_file,
    neighbor_joining, to_subby_tree, jukes_cantor_distances,
)
from jaxrate.inside import inside


def reorder_alignment_for_tree(alignment, tree_result, leaf_names):
    """Reorder alignment rows and insert gap rows for internal nodes.

    subby expects alignment rows indexed by tree node (including internal
    nodes, which get gap tokens).

    Args:
        alignment: (N, C) int32 leaf-only alignment
        tree_result: dict from neighbor_joining with 'parentIndex' and 'leaf_names'
        leaf_names: original leaf names matching alignment rows

    Returns:
        (R, C) int32 array with one row per tree node
    """
    N, C = alignment.shape
    R = len(tree_result['parentIndex'])
    gap_token = 99

    # Map original leaf names to alignment row indices
    name_to_row = {name: i for i, name in enumerate(leaf_names)}

    # Map tree leaf order to original alignment rows
    tree_leaf_idx = 0
    full_alignment = np.full((R, C), gap_token, dtype=np.int32)

    # Find leaf positions in the tree (nodes with no children)
    children_count = np.zeros(R, dtype=np.int32)
    for i in range(1, R):
        children_count[tree_result['parentIndex'][i]] += 1
    is_leaf = children_count == 0

    tree_leaf_pos = 0
    for node_idx in range(R):
        if is_leaf[node_idx]:
            leaf_name = tree_result['leaf_names'][tree_leaf_pos]
            if leaf_name in name_to_row:
                full_alignment[node_idx] = alignment[name_to_row[leaf_name]]
            tree_leaf_pos += 1

    return full_alignment


def parse_ss_cons(ss_cons):
    """Parse SS_cons annotation into base pairs.

    Handles bracket pairs (</>), uppercase/lowercase letter pairs (A/a),
    and standard bracket types.

    Returns:
        pairs: sorted list of (i, j) base pairs (0-indexed)
    """
    pairs = []
    stack_bracket = []
    stacks = {}

    for i, ch in enumerate(ss_cons):
        if ch in '<([{':
            stack_bracket.append(i)
        elif ch in '>)]}':
            if stack_bracket:
                j = stack_bracket.pop()
                pairs.append((j, i))
        elif ch.isupper() and ch.isalpha():
            if ch not in stacks:
                stacks[ch] = []
            stacks[ch].append(i)
        elif ch.islower() and ch.isalpha():
            letter = ch.upper()
            if letter in stacks and stacks[letter]:
                j = stacks[letter].pop(0)
                pairs.append((j, i))

    return sorted(pairs)


def main():
    data_dir = os.path.join(os.path.dirname(__file__), '..', 'jaxrate', 'data')

    # --- Step 1: Parse pfold.eg grammar ---
    print("=== Step 1: Parse pfold.eg grammar ===")
    pfold_path = os.path.join(data_dir, 'pfold.eg')
    xg = parse_xrate_file(pfold_path)
    grammar = xg.grammar
    print(f"Nonterminals: {[nt.name for nt in grammar.nonterminals]}")
    print(f"Rules: {len(grammar.rules)}")
    print(f"Classification: pfold is an SCFG")
    print(f"Single chains: {len(xg.get_single_models())}")
    print(f"Paired chains: {len(xg.get_paired_models())}")

    # --- Step 2: Load RF00390 alignment ---
    print("\n=== Step 2: Load RF00390 alignment ===")
    sto_path = os.path.join(data_dir, 'RF00390.sto')
    with open(sto_path) as f:
        sto_text = f.read()

    # Parse SS_cons manually (subby's parser skips GC lines)
    ss_cons = None
    for line in sto_text.splitlines():
        if line.startswith('#=GC SS_cons'):
            ss_cons = line.split()[-1]

    # Parse alignment with RNA alphabet
    sto = parse_stockholm(sto_text, alphabet=['A', 'C', 'G', 'U'])
    alignment = sto['alignment']
    leaf_names = sto['leaf_names']
    N, C = alignment.shape
    print(f"Sequences: {N}, Columns: {C}")
    print(f"Leaves: {leaf_names[:3]}...")
    print(f"SS_cons: {ss_cons}")

    pairs = parse_ss_cons(ss_cons)
    print(f"Annotated base pairs: {pairs}")

    # --- Step 3: Build NJ tree ---
    print("\n=== Step 3: Build NJ tree ===")
    D = jukes_cantor_distances(alignment, A=4)
    nj_result = neighbor_joining(D, leaf_names)
    tree = to_subby_tree(nj_result)
    R = len(nj_result['parentIndex'])
    print(f"Tree nodes: {R} ({N} leaves + {R - N} internal)")
    print(f"Newick: {nj_result['newick'][:80]}...")

    # Reorder alignment to match tree node order
    full_alignment = reorder_alignment_for_tree(alignment, nj_result, leaf_names)
    print(f"Full alignment shape: {full_alignment.shape}")

    # --- Step 4: Compute phylogenetic terminal weights ---
    print("\n=== Step 4: Compute terminal weights via subby ===")
    subby_models = xg.to_subby_models()

    # Single-column weights
    single_model = subby_models['single'][0]
    single_ll = LogLike(jnp.array(full_alignment), tree, single_model)
    print(f"Single-column log-likelihoods shape: {single_ll.shape}")
    print(f"  Mean: {float(single_ll.mean()):.4f}")
    print(f"  Range: [{float(single_ll.min()):.4f}, {float(single_ll.max()):.4f}]")

    # Paired-column weights
    paired_model_info = subby_models['paired'][0]
    paired_model = paired_model_info['model']
    A = paired_model_info['A']

    # Generate all column pairs and compute paired log-likelihoods
    tuples = all_column_ktuples(C, 2, ordered=True)
    if len(tuples) > 0:
        kt = kmer_tokenize(full_alignment, A, tuples, gap_mode='any')
        paired_ll_flat = LogLike(jnp.array(kt['alignment']), tree, paired_model)

        # Reshape into (C, C) matrix
        paired_ll_matrix = jnp.full((C, C), -1e38)
        for t_idx in range(len(tuples)):
            ci, cj = tuples[t_idx]
            paired_ll_matrix = paired_ll_matrix.at[ci, cj].set(paired_ll_flat[t_idx])

        print(f"Paired-column log-likelihoods shape: {paired_ll_matrix.shape}")
        # Show values at annotated base pairs
        for i, j in pairs[:5]:
            if i < C and j < C:
                print(f"  Pair ({i},{j}): single_i={float(single_ll[i]):.3f}, "
                      f"single_j={float(single_ll[j]):.3f}, "
                      f"paired={float(paired_ll_matrix[i, j]):.3f}")
    else:
        paired_ll_matrix = None

    # Build TerminalWeights
    tw = TerminalWeights(
        single=single_ll[None, :],  # (1, C)
        paired=paired_ll_matrix[None, :, :] if paired_ll_matrix is not None else None,
        C=C,
    )

    # --- Step 5: Run algorithms ---
    print("\n=== Step 5: Run inside and Viterbi ===")

    # Inside algorithm
    chart, log_ll = inside(grammar, tw)
    print(f"Inside log-likelihood: {float(log_ll):.4f}")

    # Viterbi
    labels, log_prob = viterbi(grammar, tw)
    print(f"Viterbi log-probability: {float(log_prob):.4f}")

    nt_names = [nt.name for nt in grammar.nonterminals]
    annotation = [nt_names[int(l)] for l in labels]

    # --- Step 6: Compare to SS_cons ---
    print("\n=== Step 6: Results ===")

    # Build display strings
    seq_ref = list(sto['leaf_names'])[0]
    seq_str = ''.join(
        ['A', 'C', 'G', 'U'][alignment[0, c]] if alignment[0, c] < 4 else '-'
        for c in range(C)
    )

    # Map annotations to structure characters
    # pfold nonterminals: pfoldS, pfoldL, pfoldB, pfoldF, pfoldF*, pfoldU, pfoldU*
    # pfoldF emits paired (< >) — structure
    # pfoldU emits unpaired (.) — loop
    # pfoldL, pfoldS, pfoldB, pfoldF*, pfoldU* are null transitions
    struct_chars = []
    for a in annotation:
        if a == 'pfoldF':
            struct_chars.append('<')
        elif a == 'pfoldU':
            struct_chars.append('.')
        else:
            struct_chars.append('_')

    print(f"Sequence:    {seq_str}")
    print(f"SS_cons:     {ss_cons}")
    print(f"Predicted:   {''.join(struct_chars)}")
    print(f"NT labels:   {' '.join(annotation)}")

    # Compute accuracy for paired/unpaired columns
    true_paired = set()
    for i, j in pairs:
        if i < C and j < C:
            true_paired.update([i, j])
    true_unpaired = set(range(C)) - true_paired
    # Dots in SS_cons that aren't letters
    ss_unpaired = {i for i, ch in enumerate(ss_cons[:C])
                   if ch in '.:_~,' or ch == '-'}

    pred_paired = {i for i, a in enumerate(annotation) if a == 'pfoldF'}
    pred_unpaired = {i for i, a in enumerate(annotation) if a == 'pfoldU'}

    print(f"\nTrue paired columns:      {sorted(true_paired)}")
    print(f"Predicted paired columns: {sorted(pred_paired)}")
    print(f"True unpaired columns:    {sorted(true_unpaired)}")
    print(f"Predicted unpaired:       {sorted(pred_unpaired)}")


if __name__ == '__main__':
    main()
