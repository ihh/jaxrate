"""Tests for wiggle output module."""

import os
import tempfile

import numpy as np
import pytest

from jaxrate.wiggle import (
    write_wiggle, write_bedgraph, write_multi_wiggle,
    posteriors_to_structure_track,
)
from jaxrate import GrammarBuilder, EmissionGroup


class TestWriteWiggle:
    def test_basic(self):
        values = np.array([0.1, 0.5, 0.9, 0.3])
        with tempfile.NamedTemporaryFile(mode='w', suffix='.wig', delete=False) as f:
            path = f.name
        try:
            write_wiggle(path, values, chrom='chr1', start=100,
                         track_name='test', description='test track')
            with open(path) as f:
                lines = f.readlines()
            assert 'track type=wiggle_0' in lines[0]
            assert 'name="test"' in lines[0]
            assert 'fixedStep chrom=chr1 start=100 step=1' in lines[1]
            assert len(lines) == 6  # header + fixedStep + 4 values
            assert float(lines[2].strip()) == pytest.approx(0.1)
            assert float(lines[5].strip()) == pytest.approx(0.3)
        finally:
            os.unlink(path)


class TestWriteBedGraph:
    def test_basic(self):
        values = np.array([0.1, 0.5, 0.9])
        with tempfile.NamedTemporaryFile(mode='w', suffix='.bedGraph', delete=False) as f:
            path = f.name
        try:
            write_bedgraph(path, values, chrom='chr1', start=10,
                           track_name='bg_test')
            with open(path) as f:
                lines = f.readlines()
            assert 'type=bedGraph' in lines[0]
            # Second line: chr1\t10\t11\t0.100000
            parts = lines[1].strip().split('\t')
            assert parts[0] == 'chr1'
            assert parts[1] == '10'
            assert parts[2] == '11'
        finally:
            os.unlink(path)


class TestMultiWiggle:
    def test_two_tracks(self):
        tracks = [
            {'name': 'track1', 'values': np.array([0.1, 0.2])},
            {'name': 'track2', 'values': np.array([0.9, 0.8])},
        ]
        with tempfile.NamedTemporaryFile(mode='w', suffix='.wig', delete=False) as f:
            path = f.name
        try:
            write_multi_wiggle(path, tracks, chrom='poliovirus')
            with open(path) as f:
                content = f.read()
            assert content.count('track type=wiggle_0') == 2
            assert 'name="track1"' in content
            assert 'name="track2"' in content
        finally:
            os.unlink(path)


class TestPosteriorsToStructureTrack:
    def test_basic(self):
        gb = GrammarBuilder()
        S = gb.add_nonterminal('S')
        F = gb.add_nonterminal('F')
        U = gb.add_nonterminal('U')

        # F has paired emission -> structural
        gb.add_rule(S, rhs=[F])
        gb.add_rule(F, rhs=[S],
                    emissions=[EmissionGroup(n_positions=2, model_index=0)])
        gb.add_rule(U, rhs=[],
                    emissions=[EmissionGroup(n_positions=1, model_index=0)])

        grammar = gb.build(start=S, n_models=1)

        # Posteriors: 4 columns, 3 nonterminals
        posteriors = np.array([
            [0.1, 0.8, 0.1],  # col 0: mostly F (structural)
            [0.9, 0.0, 0.1],  # col 1: mostly S
            [0.2, 0.7, 0.1],  # col 2: mostly F (structural)
            [0.1, 0.0, 0.9],  # col 3: mostly U
        ])

        track = posteriors_to_structure_track(posteriors, grammar)
        assert track.shape == (4,)
        assert track[0] == pytest.approx(0.8)  # F posterior
        assert track[1] == pytest.approx(0.0)
        assert track[2] == pytest.approx(0.7)
        assert track[3] == pytest.approx(0.0)
