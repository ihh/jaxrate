"""Neighbor-Joining tree construction.

Implements the Neighbor-Joining (NJ) algorithm for building phylogenetic
trees from pairwise distance matrices. Produces trees compatible with
subby's Tree format.
"""

import numpy as np
import jax.numpy as jnp


def neighbor_joining(distance_matrix, leaf_names=None):
    """Construct a tree using the Neighbor-Joining algorithm.

    Args:
        distance_matrix: (N, N) symmetric distance matrix (numpy or jax array)
        leaf_names: optional list of N leaf names

    Returns:
        dict with:
            'parentIndex': (R,) int32 array in preorder
            'distanceToParent': (R,) float64 array
            'leaf_names': list of leaf names in tree order
            'newick': Newick string representation
    """
    D = np.array(distance_matrix, dtype=np.float64)
    N = D.shape[0]
    assert D.shape == (N, N), f"Distance matrix must be square, got {D.shape}"

    if leaf_names is None:
        leaf_names = [f"leaf_{i}" for i in range(N)]
    assert len(leaf_names) == N

    if N == 1:
        return {
            'parentIndex': np.array([-1], dtype=np.int32),
            'distanceToParent': np.array([0.0]),
            'leaf_names': list(leaf_names),
            'newick': f"{leaf_names[0]};",
        }

    if N == 2:
        d = D[0, 1]
        return {
            'parentIndex': np.array([-1, 0, 0], dtype=np.int32),
            'distanceToParent': np.array([0.0, d / 2, d / 2]),
            'leaf_names': list(leaf_names),
            'newick': f"({leaf_names[0]}:{d/2},{leaf_names[1]}:{d/2});",
        }

    # NJ algorithm
    # Represent nodes as indices into a growing list
    # Active nodes are tracked in a set
    nodes = list(range(N))  # initial node indices
    node_subtrees = {i: leaf_names[i] for i in range(N)}  # newick subtrees
    dist = {(i, j): D[i, j] for i in range(N) for j in range(N) if i != j}
    next_node = N

    # Children and branch lengths for tree construction
    children = {}  # node -> [(child, branch_length), ...]

    active = set(range(N))

    while len(active) > 2:
        active_list = sorted(active)
        n = len(active_list)

        # Compute r_i = sum of distances from i to all other active nodes
        r = {}
        for i in active_list:
            r[i] = sum(dist.get((i, j), 0.0) for j in active_list if j != i)

        # Find pair (i, j) minimizing Q(i,j) = (n-2)*d(i,j) - r(i) - r(j)
        best_q = float('inf')
        best_i, best_j = active_list[0], active_list[1]
        for idx_a in range(n):
            for idx_b in range(idx_a + 1, n):
                i, j = active_list[idx_a], active_list[idx_b]
                q = (n - 2) * dist[(i, j)] - r[i] - r[j]
                if q < best_q:
                    best_q = q
                    best_i, best_j = i, j

        # Create new internal node
        u = next_node
        next_node += 1

        # Branch lengths from new node to i, j
        d_ij = dist[(best_i, best_j)]
        if n > 2:
            d_iu = 0.5 * d_ij + (r[best_i] - r[best_j]) / (2 * (n - 2))
            d_ju = d_ij - d_iu
        else:
            d_iu = d_ij / 2
            d_ju = d_ij / 2

        # Clamp negative branch lengths to 0
        d_iu = max(d_iu, 0.0)
        d_ju = max(d_ju, 0.0)

        children[u] = [(best_i, d_iu), (best_j, d_ju)]

        # Build newick subtree for new node
        sub_i = node_subtrees[best_i]
        sub_j = node_subtrees[best_j]
        node_subtrees[u] = f"({sub_i}:{d_iu:.6f},{sub_j}:{d_ju:.6f})"

        # Update distances: d(u, k) for all active k != i, j
        for k in active_list:
            if k == best_i or k == best_j:
                continue
            d_uk = 0.5 * (dist[(best_i, k)] + dist[(best_j, k)] - d_ij)
            d_uk = max(d_uk, 0.0)
            dist[(u, k)] = d_uk
            dist[(k, u)] = d_uk

        # Remove i, j from active; add u
        active.discard(best_i)
        active.discard(best_j)
        active.add(u)

    # Final two nodes: connect them
    remaining = sorted(active)
    assert len(remaining) == 2
    a, b = remaining
    root = next_node
    d_ab = dist.get((a, b), 0.0)
    children[root] = [(a, d_ab / 2), (b, d_ab / 2)]

    newick = f"({node_subtrees[a]}:{d_ab/2:.6f},{node_subtrees[b]}:{d_ab/2:.6f});"

    # Convert to preorder arrays
    parent_index = []
    distance_to_parent = []
    tree_leaf_names = []
    old_to_new = {}

    def dfs(old_idx, parent_new):
        new_idx = len(parent_index)
        old_to_new[old_idx] = new_idx
        parent_index.append(parent_new)
        distance_to_parent.append(0.0 if parent_new < 0 else -1.0)
        if old_idx in children:
            for child, bl in children[old_idx]:
                child_new = len(parent_index)
                dfs(child, new_idx)
                distance_to_parent[child_new] = bl
        else:
            # Leaf node
            tree_leaf_names.append(leaf_names[old_idx])

    dfs(root, -1)

    return {
        'parentIndex': np.array(parent_index, dtype=np.int32),
        'distanceToParent': np.array(distance_to_parent),
        'leaf_names': tree_leaf_names,
        'newick': newick,
    }


def to_subby_tree(nj_result):
    """Convert NJ result to subby Tree NamedTuple.

    Args:
        nj_result: dict from neighbor_joining()

    Returns:
        subby Tree NamedTuple
    """
    from subby.jax.types import Tree
    return Tree(
        parentIndex=jnp.array(nj_result['parentIndex']),
        distanceToParent=jnp.array(nj_result['distanceToParent']),
    )


def hamming_distances(alignment, normalize=True):
    """Compute pairwise Hamming distances from an alignment.

    Gaps are excluded from comparison (only positions where both
    sequences have non-gap characters are counted).

    Args:
        alignment: (N, C) int32 token-encoded alignment
        normalize: if True, return fraction of differing positions;
                   if False, return count of differing positions

    Returns:
        (N, N) float64 distance matrix
    """
    alignment = np.array(alignment, dtype=np.int32)
    N, C = alignment.shape
    # Assume gap tokens are >= alphabet size (typically 4 for RNA)
    # subby convention: gap_idx = len(alphabet) + 1
    # We treat any token >= 4 as a gap for RNA, or >= 20 for protein
    # Use a generous threshold
    gap_threshold = 4  # tokens 0-3 are valid for RNA

    D = np.zeros((N, N), dtype=np.float64)
    for i in range(N):
        for j in range(i + 1, N):
            valid = (alignment[i] < gap_threshold) & (alignment[j] < gap_threshold)
            n_valid = valid.sum()
            if n_valid == 0:
                D[i, j] = D[j, i] = 1.0  # max distance for no overlap
            else:
                n_diff = ((alignment[i] != alignment[j]) & valid).sum()
                if normalize:
                    D[i, j] = D[j, i] = n_diff / n_valid
                else:
                    D[i, j] = D[j, i] = float(n_diff)
    return D


def jukes_cantor_distances(alignment, A=4):
    """Compute Jukes-Cantor corrected pairwise distances.

    d_JC = -(A-1)/A * ln(1 - A/(A-1) * p)

    where p is the fraction of differing sites.

    Args:
        alignment: (N, C) int32 token-encoded alignment
        A: alphabet size (4 for nucleotide, 20 for protein)

    Returns:
        (N, N) float64 distance matrix
    """
    p = hamming_distances(alignment, normalize=True)
    # JC correction
    correction_factor = A / (A - 1.0)
    # Clamp p to avoid log of negative number
    p_clamped = np.minimum(p, (A - 1.0) / A - 1e-10)
    d_jc = -(A - 1.0) / A * np.log(1.0 - correction_factor * p_clamped)
    np.fill_diagonal(d_jc, 0.0)
    return d_jc
