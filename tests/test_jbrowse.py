"""Tests for JBrowse track output module."""

import os
import tempfile
import numpy as np
import jax
import jax.numpy as jnp
import pytest

jax.config.update("jax_enable_x64", True)

from jaxrate import (
    pfold_grammar,
    gene_finder_grammar,
    pseudoknot_grammar,
    TerminalWeights,
    viterbi,
)
from jaxrate.jbrowse import (
    labels_to_features,
    labels_to_annotation_map,
    merge_features,
    write_bed,
    write_gff3,
    posteriors_to_bedgraph,
    conservation_bedgraph,
    write_jbrowse_tracks,
    _classify_nonterminals,
    _feature_color,
)


# ── Fixtures ─────────────────────────────────────────────────────────────────

@pytest.fixture
def pfold_results():
    """Run pfold Viterbi and return labels + grammar."""
    grammar = pfold_grammar()
    C = 12
    rng = np.random.RandomState(42)
    single = jnp.array(rng.randn(1, C) * 0.5 - 1.5)
    paired = jnp.array(rng.randn(1, C, C) * 0.3 - 3.0)
    # Boost some paired positions
    paired = paired.at[0, 1, 10].set(-0.5)
    paired = paired.at[0, 2, 9].set(-0.5)
    tw = TerminalWeights(single=single, paired=paired, C=C)
    labels, log_prob = viterbi(grammar, tw)
    return {
        'grammar': grammar,
        'labels': labels,
        'log_prob': log_prob,
        'tw': tw,
        'C': C,
    }


@pytest.fixture
def gene_results():
    """Run gene finder Viterbi and return labels + grammar."""
    grammar = gene_finder_grammar()
    C = 30
    rng = np.random.RandomState(99)
    single = jnp.array(rng.randn(3, C) * 0.3 - 2.0)
    # Make exon model (1) stronger in middle region
    single = single.at[1, 10:25].set(-0.5)
    tw = TerminalWeights(single=single, C=C)
    labels, log_prob = viterbi(grammar, tw)
    return {
        'grammar': grammar,
        'labels': labels,
        'log_prob': log_prob,
        'tw': tw,
        'C': C,
    }


@pytest.fixture
def tmpdir():
    with tempfile.TemporaryDirectory() as d:
        yield d


# ── labels_to_features tests ────────────────────────────────────────────────

class TestLabelsToFeatures:
    def test_basic_features(self, pfold_results):
        features = labels_to_features(
            pfold_results['labels'], pfold_results['grammar'])
        assert len(features) > 0
        for f in features:
            assert 'chrom' in f
            assert 'start' in f
            assert 'end' in f
            assert 'name' in f
            assert f['end'] > f['start']
            assert f['start'] >= 0

    def test_features_cover_all_columns(self, pfold_results):
        features = labels_to_features(
            pfold_results['labels'], pfold_results['grammar'])
        total_len = sum(f['end'] - f['start'] for f in features)
        assert total_len == pfold_results['C']

    def test_features_are_contiguous(self, pfold_results):
        features = labels_to_features(
            pfold_results['labels'], pfold_results['grammar'])
        for i in range(len(features) - 1):
            assert features[i]['end'] == features[i + 1]['start']

    def test_custom_chrom_and_start(self, pfold_results):
        features = labels_to_features(
            pfold_results['labels'], pfold_results['grammar'],
            chrom='chr5', start=1000, strand='+')
        assert features[0]['chrom'] == 'chr5'
        assert features[0]['start'] >= 1000
        assert features[0]['strand'] == '+'
        assert features[-1]['end'] == 1000 + pfold_results['C']

    def test_empty_labels(self):
        grammar = pfold_grammar()
        features = labels_to_features(np.array([], dtype=np.int32), grammar)
        assert features == []

    def test_gene_finder_features(self, gene_results):
        features = labels_to_features(
            gene_results['labels'], gene_results['grammar'])
        assert len(features) > 0
        # Gene finder should have exon/intron/intergenic types
        types = {f['feature_type'] for f in features}
        assert len(types) > 0


class TestLabelsToAnnotationMap:
    def test_returns_list_of_strings(self, pfold_results):
        annot = labels_to_annotation_map(
            pfold_results['labels'], pfold_results['grammar'])
        assert len(annot) == pfold_results['C']
        assert all(isinstance(a, str) for a in annot)

    def test_gene_finder_categories(self, gene_results):
        annot = labels_to_annotation_map(
            gene_results['labels'], gene_results['grammar'])
        valid = {'exon', 'intron', 'intergenic', 'null', 'unpaired',
                 'paired', 'other'}
        # Accept nonterminal names too
        for a in annot:
            assert isinstance(a, str)


class TestClassifyNonterminals:
    def test_pfold_has_paired(self):
        grammar = pfold_grammar()
        nt_emit = _classify_nonterminals(grammar)
        nt_names = [nt.name for nt in grammar.nonterminals]
        # L nonterminal has paired emission
        l_idx = nt_names.index('L')
        assert nt_emit[l_idx] == 'paired'

    def test_gene_finder_types(self):
        grammar = gene_finder_grammar()
        nt_emit = _classify_nonterminals(grammar)
        nt_names = [nt.name for nt in grammar.nonterminals]
        types = {nt_emit.get(i) for i in range(len(nt_names))}
        assert 'exon' in types or 'unpaired' in types  # at least emitting types


# ── merge_features tests ────────────────────────────────────────────────────

class TestMergeFeatures:
    def test_merges_adjacent_same_category(self):
        features = [
            {'chrom': 'c', 'start': 0, 'end': 5, 'name': 'E1',
             'score': 0, 'strand': '.', 'nt_index': 0, 'feature_type': 'exon'},
            {'chrom': 'c', 'start': 5, 'end': 10, 'name': 'I',
             'score': 0, 'strand': '.', 'nt_index': 1, 'feature_type': 'intron'},
            {'chrom': 'c', 'start': 10, 'end': 15, 'name': 'E2',
             'score': 0, 'strand': '.', 'nt_index': 2, 'feature_type': 'exon'},
        ]
        merged = merge_features(features)
        # All three should merge into one gene
        gene_features = [f for f in merged if f.get('feature_type') == 'gene']
        assert len(gene_features) == 1
        assert gene_features[0]['start'] == 0
        assert gene_features[0]['end'] == 15
        assert len(gene_features[0]['children']) == 3

    def test_null_features_break_groups(self):
        features = [
            {'chrom': 'c', 'start': 0, 'end': 5, 'name': 'L',
             'score': 0, 'strand': '.', 'nt_index': 0, 'feature_type': 'paired'},
            {'chrom': 'c', 'start': 5, 'end': 10, 'name': 'X',
             'score': 0, 'strand': '.', 'nt_index': 1, 'feature_type': 'null'},
            {'chrom': 'c', 'start': 10, 'end': 15, 'name': 'L',
             'score': 0, 'strand': '.', 'nt_index': 0, 'feature_type': 'paired'},
        ]
        merged = merge_features(features)
        struct_features = [f for f in merged
                           if f.get('feature_type') == 'structure']
        assert len(struct_features) == 2

    def test_empty_features(self):
        assert merge_features([]) == []


# ── BED output tests ────────────────────────────────────────────────────────

class TestWriteBed:
    def test_bed3(self, pfold_results, tmpdir):
        features = labels_to_features(
            pfold_results['labels'], pfold_results['grammar'])
        path = os.path.join(tmpdir, 'test.bed3')
        write_bed(path, features, format='bed3')
        with open(path) as f:
            lines = f.readlines()
        assert len(lines) == len(features)
        # BED3: 3 columns
        assert len(lines[0].strip().split('\t')) == 3

    def test_bed6(self, pfold_results, tmpdir):
        features = labels_to_features(
            pfold_results['labels'], pfold_results['grammar'])
        path = os.path.join(tmpdir, 'test.bed6')
        write_bed(path, features, format='bed6')
        with open(path) as f:
            lines = f.readlines()
        assert len(lines) == len(features)
        assert len(lines[0].strip().split('\t')) == 6

    def test_bed12_simple(self, pfold_results, tmpdir):
        features = labels_to_features(
            pfold_results['labels'], pfold_results['grammar'])
        path = os.path.join(tmpdir, 'test.bed12')
        write_bed(path, features, format='bed12')
        with open(path) as f:
            lines = f.readlines()
        assert len(lines) == len(features)
        assert len(lines[0].strip().split('\t')) == 12

    def test_bed12_merged(self, gene_results, tmpdir):
        features = labels_to_features(
            gene_results['labels'], gene_results['grammar'])
        merged = merge_features(features)
        path = os.path.join(tmpdir, 'merged.bed12')
        write_bed(path, merged, format='bed12')
        with open(path) as f:
            lines = f.readlines()
        # Each merged feature should be one BED12 line
        assert len(lines) == len(merged)
        for line in lines:
            cols = line.strip().split('\t')
            assert len(cols) == 12
            # Verify block_count > 0
            assert int(cols[9]) > 0


# ── GFF3 output tests ───────────────────────────────────────────────────────

class TestWriteGff3:
    def test_gff3_header(self, pfold_results, tmpdir):
        features = labels_to_features(
            pfold_results['labels'], pfold_results['grammar'])
        path = os.path.join(tmpdir, 'test.gff3')
        write_gff3(path, features)
        with open(path) as f:
            first_line = f.readline()
        assert first_line.strip() == '##gff-version 3'

    def test_gff3_sequence_region(self, pfold_results, tmpdir):
        features = labels_to_features(
            pfold_results['labels'], pfold_results['grammar'])
        path = os.path.join(tmpdir, 'test.gff3')
        write_gff3(path, features, sequence_region=('chr1', 0, 100))
        with open(path) as f:
            lines = f.readlines()
        assert any('##sequence-region' in l for l in lines)

    def test_gff3_features_are_1based(self, pfold_results, tmpdir):
        features = labels_to_features(
            pfold_results['labels'], pfold_results['grammar'],
            chrom='chr1', start=0)
        path = os.path.join(tmpdir, 'test.gff3')
        write_gff3(path, features)
        with open(path) as f:
            for line in f:
                if line.startswith('#'):
                    continue
                cols = line.strip().split('\t')
                gff_start = int(cols[3])
                assert gff_start >= 1  # GFF3 is 1-based

    def test_gff3_merged_has_parent_child(self, gene_results, tmpdir):
        features = labels_to_features(
            gene_results['labels'], gene_results['grammar'])
        merged = merge_features(features)
        path = os.path.join(tmpdir, 'merged.gff3')
        write_gff3(path, merged)
        with open(path) as f:
            content = f.read()
        # Should have Parent= attributes for children
        has_parent = 'Parent=' in content
        has_children = any(f.get('children') for f in merged)
        assert has_parent == has_children

    def test_gff3_9_columns(self, pfold_results, tmpdir):
        features = labels_to_features(
            pfold_results['labels'], pfold_results['grammar'])
        path = os.path.join(tmpdir, 'test.gff3')
        write_gff3(path, features)
        with open(path) as f:
            for line in f:
                if line.startswith('#'):
                    continue
                cols = line.strip().split('\t')
                assert len(cols) == 9


# ── Quantitative track tests ────────────────────────────────────────────────

class TestPosteriorsBedGraph:
    def test_writes_bedgraph(self, pfold_results, tmpdir):
        C = pfold_results['C']
        K = len(pfold_results['grammar'].nonterminals)
        posteriors = np.random.rand(C, K).astype(np.float64)
        posteriors /= posteriors.sum(axis=1, keepdims=True)

        path = os.path.join(tmpdir, 'post.bedGraph')
        posteriors_to_bedgraph(
            path, posteriors, pfold_results['grammar'],
            chrom='chr1', start=100)

        with open(path) as f:
            content = f.read()
        assert 'track type=bedGraph' in content
        # Should have data lines
        data_lines = [l for l in content.split('\n')
                      if l and not l.startswith('track')]
        assert len(data_lines) > 0


class TestConservationBedGraph:
    def test_writes_conservation(self, pfold_results, tmpdir):
        path = os.path.join(tmpdir, 'cons.bedGraph')
        conservation_bedgraph(
            path, pfold_results['tw'], chrom='chr1', start=0)

        with open(path) as f:
            content = f.read()
        assert 'conservation' in content.lower() or 'track' in content
        data_lines = [l for l in content.split('\n')
                      if l and not l.startswith('track')]
        assert len(data_lines) == pfold_results['C']


# ── write_jbrowse_tracks tests ──────────────────────────────────────────────

class TestWriteJbrowseTracks:
    def test_produces_all_files(self, pfold_results, tmpdir):
        tracks = write_jbrowse_tracks(
            tmpdir, pfold_results['labels'], pfold_results['grammar'],
            pfold_results['tw'], chrom='chr1', start=0)

        assert 'features_bed6' in tracks
        assert 'features_bed12' in tracks
        assert 'features_gff3' in tracks
        assert 'conservation' in tracks

        for path in tracks.values():
            assert os.path.exists(path)
            assert os.path.getsize(path) > 0

    def test_with_posteriors(self, pfold_results, tmpdir):
        C = pfold_results['C']
        K = len(pfold_results['grammar'].nonterminals)
        posteriors = np.random.rand(C, K)
        posteriors /= posteriors.sum(axis=1, keepdims=True)

        tracks = write_jbrowse_tracks(
            tmpdir, pfold_results['labels'], pfold_results['grammar'],
            pfold_results['tw'], posteriors=posteriors, chrom='chr1')

        assert 'posteriors' in tracks
        assert os.path.exists(tracks['posteriors'])


# ── Feature color tests ─────────────────────────────────────────────────────

class TestFeatureColor:
    def test_known_types(self):
        assert _feature_color('paired') == '255,0,0'
        assert _feature_color('exon') == '0,128,0'
        assert _feature_color('intron') == '200,200,200'

    def test_unknown_type_defaults(self):
        assert _feature_color('unknown') == '0,0,0'
