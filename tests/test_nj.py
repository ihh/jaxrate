"""Tests for Neighbor-Joining tree construction."""

import numpy as np
import jax.numpy as jnp
import pytest

from jaxrate.nj import (
    neighbor_joining,
    to_subby_tree,
    hamming_distances,
    jukes_cantor_distances,
)


class TestNeighborJoining:
    def test_two_taxa(self):
        D = np.array([[0, 2], [2, 0]], dtype=np.float64)
        result = neighbor_joining(D, ['A', 'B'])
        tree = to_subby_tree(result)
        assert len(tree.parentIndex) == 3  # root + 2 leaves
        assert tree.parentIndex[0] == -1  # root
        np.testing.assert_allclose(
            tree.distanceToParent[1] + tree.distanceToParent[2], 2.0, atol=1e-10)

    def test_three_taxa(self):
        # Distances: A-B=2, A-C=4, B-C=4
        D = np.array([
            [0, 2, 4],
            [2, 0, 4],
            [4, 4, 0],
        ], dtype=np.float64)
        result = neighbor_joining(D, ['A', 'B', 'C'])
        tree = to_subby_tree(result)
        assert tree.parentIndex[0] == -1
        assert len(result['leaf_names']) == 3

    def test_four_taxa_symmetric(self):
        # Symmetric caterpillar tree: ((A,B),(C,D))
        # True distances: A-B=2, C-D=2, A-C=A-D=B-C=B-D=4
        D = np.array([
            [0, 2, 4, 4],
            [2, 0, 4, 4],
            [4, 4, 0, 2],
            [4, 4, 2, 0],
        ], dtype=np.float64)
        result = neighbor_joining(D, ['A', 'B', 'C', 'D'])
        tree = to_subby_tree(result)
        assert tree.parentIndex[0] == -1
        # Total branch length should be consistent
        total_bl = float(tree.distanceToParent.sum())
        assert total_bl > 0

    def test_preorder_invariant(self):
        """Parent index should always be less than child index in preorder."""
        D = np.array([
            [0, 3, 5, 7],
            [3, 0, 6, 8],
            [5, 6, 0, 4],
            [7, 8, 4, 0],
        ], dtype=np.float64)
        result = neighbor_joining(D)
        pi = result['parentIndex']
        for i in range(1, len(pi)):
            assert pi[i] < i, f"Node {i} has parent {pi[i]} >= {i}"

    def test_single_taxon(self):
        D = np.array([[0.0]])
        result = neighbor_joining(D, ['solo'])
        assert len(result['parentIndex']) == 1
        assert result['leaf_names'] == ['solo']

    def test_newick_output(self):
        D = np.array([[0, 2], [2, 0]], dtype=np.float64)
        result = neighbor_joining(D, ['A', 'B'])
        assert 'newick' in result
        assert 'A' in result['newick']
        assert 'B' in result['newick']
        assert result['newick'].endswith(';')

    def test_nonnegative_branch_lengths(self):
        """Branch lengths should never be negative."""
        np.random.seed(42)
        N = 6
        D = np.random.rand(N, N) * 5
        D = (D + D.T) / 2
        np.fill_diagonal(D, 0)
        result = neighbor_joining(D)
        assert all(d >= 0 for d in result['distanceToParent'])

    def test_subby_tree_format(self):
        D = np.array([
            [0, 1, 2],
            [1, 0, 2],
            [2, 2, 0],
        ], dtype=np.float64)
        result = neighbor_joining(D, ['x', 'y', 'z'])
        tree = to_subby_tree(result)
        assert hasattr(tree, 'parentIndex')
        assert hasattr(tree, 'distanceToParent')
        assert tree.parentIndex.dtype == jnp.int32
        assert tree.distanceToParent.shape == tree.parentIndex.shape


class TestDistanceMatrices:
    def test_hamming_identical(self):
        alignment = np.array([[0, 1, 2, 3], [0, 1, 2, 3]], dtype=np.int32)
        D = hamming_distances(alignment)
        np.testing.assert_allclose(D[0, 1], 0.0)
        np.testing.assert_allclose(D[1, 0], 0.0)

    def test_hamming_all_different(self):
        alignment = np.array([[0, 0, 0, 0], [1, 1, 1, 1]], dtype=np.int32)
        D = hamming_distances(alignment)
        np.testing.assert_allclose(D[0, 1], 1.0)

    def test_hamming_half_different(self):
        alignment = np.array([[0, 0, 1, 1], [0, 0, 2, 2]], dtype=np.int32)
        D = hamming_distances(alignment)
        np.testing.assert_allclose(D[0, 1], 0.5)

    def test_hamming_symmetric(self):
        np.random.seed(123)
        alignment = np.random.randint(0, 4, size=(5, 20), dtype=np.int32)
        D = hamming_distances(alignment)
        np.testing.assert_allclose(D, D.T)

    def test_hamming_with_gaps(self):
        # Gap tokens >= 4 should be excluded
        alignment = np.array([
            [0, 1, 99, 2],  # gap at position 2
            [0, 2, 99, 2],  # gap at position 2
        ], dtype=np.int32)
        D = hamming_distances(alignment)
        # Only positions 0, 1, 3 are compared. Diff at position 1.
        np.testing.assert_allclose(D[0, 1], 1.0 / 3.0, atol=1e-10)

    def test_jukes_cantor_zero_distance(self):
        alignment = np.array([[0, 1, 2, 3], [0, 1, 2, 3]], dtype=np.int32)
        D = jukes_cantor_distances(alignment)
        np.testing.assert_allclose(D[0, 1], 0.0)

    def test_jukes_cantor_positive(self):
        alignment = np.array([[0, 0, 0, 0], [0, 0, 1, 1]], dtype=np.int32)
        D = jukes_cantor_distances(alignment)
        assert D[0, 1] > 0.5  # JC correction inflates distances

    def test_jukes_cantor_symmetric(self):
        np.random.seed(42)
        alignment = np.random.randint(0, 4, size=(4, 30), dtype=np.int32)
        D = jukes_cantor_distances(alignment)
        np.testing.assert_allclose(D, D.T)


class TestNJWithAlignment:
    def test_rf00390_alignment(self):
        """Build NJ tree from RF00390 alignment."""
        import os
        data_dir = os.path.join(os.path.dirname(__file__), '..', 'jaxrate', 'data')
        sto_path = os.path.join(data_dir, 'RF00390.sto')

        from subby.formats import parse_stockholm
        with open(sto_path) as f:
            sto = parse_stockholm(f.read(), alphabet=['A', 'C', 'G', 'U'])

        alignment = sto['alignment']
        D = jukes_cantor_distances(alignment, A=4)
        result = neighbor_joining(D, sto['leaf_names'])
        tree = to_subby_tree(result)

        N = alignment.shape[0]
        assert len(result['leaf_names']) == N
        assert tree.parentIndex[0] == -1
        # Tree should have 2*N - 1 nodes
        assert len(tree.parentIndex) == 2 * N - 1
