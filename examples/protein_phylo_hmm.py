"""Phylo-HMM with per-state rate matrices: construction, fitting, and annotation.

Demonstrates:
1. Building a phylo-HMM with different substitution rate matrices per state
2. Setting up a PhyloModel from an alignment + tree + rate matrices
3. Running integrated EM training (jointly fitting grammar weights + rates)
4. Using fit_* flags to control what gets optimized
5. Comparing Viterbi annotation before and after training

The model: a 3-state HMM for conservation analysis
  - State 0 ('conserved'):  slow evolution  (low substitution rate)
  - State 1 ('moderate'):   moderate evolution
  - State 2 ('variable'):   fast evolution  (high substitution rate)

Each state has its own rate matrix. EM training fits both the transition
probabilities and the per-state rate matrices to the data.

This example uses DNA (A=4) for clear rate matrix estimation. The same
approach works for proteins (A=20) with sufficient data.
"""

import math
import numpy as np
import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

from jaxrate import (
    GrammarBuilder, EmissionGroup,
    PhyloModel, phylo_train,
    compile_grammar, viterbi, inside,
)
from jaxrate.hmm import hmm_posteriors


def build_conservation_hmm(A=4):
    """Build a 3-state conservation HMM.

    States:
        0 (conserved): slow-evolving sites, model 0
        1 (moderate):  moderately-evolving sites, model 1
        2 (variable):  fast-evolving sites, model 2

    Args:
        A: alphabet size (4 for DNA, 20 for protein)

    Returns:
        grammar, chains
    """
    gb = GrammarBuilder()
    CONS = gb.add_nonterminal('conserved')
    MOD  = gb.add_nonterminal('moderate')
    VAR  = gb.add_nonterminal('variable')

    # Transition probabilities (will be trained)
    for src, m_idx in [(CONS, 0), (MOD, 1), (VAR, 2)]:
        for dst in [CONS, MOD, VAR]:
            p = 0.90 if dst == src else 0.04
            gb.add_rule(src, rhs=[dst], emissions=[EmissionGroup(1, m_idx)],
                        log_weight=math.log(p))
        gb.add_rule(src, rhs=[], emissions=[EmissionGroup(1, m_idx)],
                    log_weight=math.log(0.02))

    grammar = gb.build(start=CONS, n_models=3)

    # Per-state JC rate matrices at different scales
    base_rate = 1.0 / (A - 1)
    def jc_Q(scale):
        Q = np.full((A, A), base_rate * scale, dtype=np.float64)
        np.fill_diagonal(Q, 0.0)
        np.fill_diagonal(Q, -Q.sum(axis=1))
        return Q

    chains = [
        {'rate_matrix': jc_Q(0.3),  'pi': np.full(A, 1.0/A)},  # conserved
        {'rate_matrix': jc_Q(1.0),  'pi': np.full(A, 1.0/A)},  # moderate
        {'rate_matrix': jc_Q(5.0),  'pi': np.full(A, 1.0/A)},  # variable
    ]

    return grammar, chains


def simulate_alignment(A=4, n_leaves=10, C=150, seed=42):
    """Generate a synthetic alignment with conserved/moderate/variable regions.

    Returns:
        alignment: (R, C) int32
        tree: subby Tree
        true_states: (C,) ground-truth state labels
    """
    np.random.seed(seed)
    gap = A + 1

    R = 1 + n_leaves
    alignment = np.full((R, C), gap, dtype=np.int32)

    ancestor = np.random.randint(0, A, size=C)

    from subby.jax.types import Tree
    parent = np.full(R, -1, dtype=np.int32)
    dist = np.zeros(R, dtype=np.float64)
    for i in range(1, R):
        parent[i] = 0
        dist[i] = 0.5
    tree = Tree(parentIndex=jnp.array(parent), distanceToParent=jnp.array(dist))

    # Three regions of equal size
    region_size = C // 3
    sub_rates = [0.02, 0.15, 0.60]  # per-site substitution probability
    regions = [(0, region_size, 0),
               (region_size, 2 * region_size, 1),
               (2 * region_size, C, 2)]

    for leaf in range(1, R):
        alignment[leaf] = ancestor.copy()
        for start, end, region in regions:
            rate = sub_rates[region]
            for c in range(start, end):
                if np.random.random() < rate:
                    new_aa = np.random.randint(0, A - 1)
                    if new_aa >= alignment[leaf, c]:
                        new_aa += 1
                    alignment[leaf, c] = new_aa

    true_states = np.zeros(C, dtype=np.int32)
    true_states[region_size:2 * region_size] = 1
    true_states[2 * region_size:] = 2

    return alignment, tree, true_states


def main():
    A = 4  # DNA alphabet
    print("=" * 70)
    print("Phylo-HMM for Conservation Analysis (DNA)")
    print("=" * 70)

    # --- Step 1: Build grammar and substitution models ---
    print("\n--- Step 1: Build 3-state conservation HMM ---")
    grammar, chains = build_conservation_hmm(A=A)
    print(f"States: {[nt.name for nt in grammar.nonterminals]}")
    print(f"Rules:  {len(grammar.rules)}")
    print(f"Models: {grammar.n_models} (one rate matrix per state)")
    for i, c in enumerate(chains):
        diag = -c['rate_matrix'].diagonal().mean()
        print(f"  Model {i}: mean rate = {diag:.2f}")

    # --- Step 2: Simulate alignment ---
    print("\n--- Step 2: Simulate alignment ---")
    alignment, tree, true_states = simulate_alignment(A=A)
    R, C = alignment.shape
    print(f"Alignment: {R} rows x {C} columns")
    print(f"Tree: star topology, {R-1} leaves, branch length 0.5")
    region_size = C // 3
    print(f"Ground truth: conserved(0-{region_size-1}), "
          f"moderate({region_size}-{2*region_size-1}), "
          f"variable({2*region_size}-{C-1})")

    # --- Step 3: Create PhyloModel ---
    print("\n--- Step 3: Create PhyloModel ---")
    pm = PhyloModel(grammar, alignment, tree, chains)
    tw = pm.terminal_weights()
    print(f"Terminal weights: single shape = {tw.single.shape}")

    labels_before, lp_before = viterbi(grammar, tw)
    _, ll_before = inside(grammar, tw)

    print(f"\nBefore training:")
    print(f"  Log-likelihood: {float(ll_before):.2f}")
    _print_annotation(labels_before, true_states, grammar)

    # --- Step 4: Train — fit everything ---
    print("\n--- Step 4: Integrated EM training (rules + rates + pi) ---")
    history = phylo_train(pm, n_iterations=20, convergence_tol=1e-4,
                          pseudocounts=1e-4)
    print(f"Iterations: {len(history)}")
    print(f"Log-likelihood trajectory:")
    for it, ll in history[:5]:
        print(f"  Iteration {it}: {ll:.2f}")
    if len(history) > 5:
        print(f"  ...")
        print(f"  Iteration {history[-1][0]}: {history[-1][1]:.2f}")

    print(f"\nFitted rate matrices (mean diagonal rate):")
    for i, c in enumerate(pm.chains):
        diag = -c['rate_matrix'].diagonal().mean()
        print(f"  Model {i} ({grammar.nonterminals[i].name}): {diag:.4f}")

    tw = pm.terminal_weights()
    labels_after, lp_after = viterbi(pm.grammar, tw)

    print(f"\nAfter training:")
    print(f"  Log-likelihood: {history[-1][1]:.2f}")
    _print_annotation(labels_after, true_states, pm.grammar)

    # --- Step 5: Demonstrate fit_* flags ---
    print("\n--- Step 5: Selective training (rates only, fixed grammar) ---")
    grammar2, chains2 = build_conservation_hmm(A=A)
    pm2 = PhyloModel(grammar2, alignment, tree, chains2)

    old_weights = [r.log_weight for r in grammar2.rules]
    history2 = phylo_train(pm2, n_iterations=10,
                           fit_rules=False, fit_rates=True, fit_pi=True,
                           pseudocounts=1e-4)

    new_weights = [r.log_weight for r in pm2.grammar.rules]
    weights_changed = any(abs(o - n) > 1e-10
                          for o, n in zip(old_weights, new_weights))
    print(f"Grammar weights changed: {weights_changed}")  # should be False
    print(f"Fitted rate matrices:")
    for i, c in enumerate(pm2.chains):
        diag = -c['rate_matrix'].diagonal().mean()
        print(f"  Model {i}: mean rate = {diag:.4f}")


def _print_annotation(labels, true_states, grammar):
    """Print Viterbi annotation vs. ground truth."""
    C = len(labels)
    state_map = {0: 'C', 1: 'M', 2: 'V'}
    true_str = ''.join(state_map[int(s)] for s in true_states)
    pred_str = ''.join(state_map.get(int(l), '?') for l in labels)

    correct = sum(1 for i in range(C) if int(labels[i]) == int(true_states[i]))
    accuracy = correct / C

    # Show first 50 columns
    show = min(C, 50)
    print(f"  True:      {true_str[:show]}{'...' if C > show else ''}")
    print(f"  Predicted: {pred_str[:show]}{'...' if C > show else ''}")
    print(f"  Accuracy:  {accuracy:.1%} ({correct}/{C})")


if __name__ == '__main__':
    main()
