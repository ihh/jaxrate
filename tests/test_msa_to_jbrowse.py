"""Tests for the MSA-to-JBrowse pipeline module."""

import os
import tempfile
import numpy as np
import jax
import jax.numpy as jnp
import pytest

jax.config.update("jax_enable_x64", True)

from jaxrate.msa_to_jbrowse import (
    load_msa,
    build_tree,
    get_grammar_for_analysis,
    msa_to_jbrowse,
    _detect_format,
    _build_token_map,
    compute_terminal_weights,
)


# ── Fixtures ─────────────────────────────────────────────────────────────────

@pytest.fixture
def data_dir():
    return os.path.join(os.path.dirname(__file__), '..', 'jaxrate', 'data')


@pytest.fixture
def rf00390_path(data_dir):
    return os.path.join(data_dir, 'RF00390.sto')


@pytest.fixture
def tmpdir():
    with tempfile.TemporaryDirectory() as d:
        yield d


@pytest.fixture
def fasta_msa(tmpdir):
    """Create a simple FASTA alignment file."""
    path = os.path.join(tmpdir, 'test.fa')
    with open(path, 'w') as f:
        f.write('>seq1\nACGUACGUACGU\n')
        f.write('>seq2\nACGUACGU-CGU\n')
        f.write('>seq3\nACGU-CGUACGU\n')
    return path


@pytest.fixture
def maf_msa(tmpdir):
    """Create a simple MAF alignment file."""
    path = os.path.join(tmpdir, 'test.maf')
    with open(path, 'w') as f:
        f.write('##maf version=1\n\n')
        f.write('a score=0\n')
        f.write('s species1.chr1 0 12 + 1000 ACGUACGUACGU\n')
        f.write('s species2.chr1 0 11 + 900  ACGUACGU-CGU\n')
        f.write('s species3.chr1 0 11 + 800  ACGU-CGUACGU\n')
        f.write('\n')
    return path


# ── Format detection tests ──────────────────────────────────────────────────

class TestDetectFormat:
    def test_stockholm_by_extension(self, tmpdir):
        path = os.path.join(tmpdir, 'test.sto')
        with open(path, 'w') as f:
            f.write('# STOCKHOLM 1.0\n')
        assert _detect_format(path) == 'stockholm'

    def test_fasta_by_extension(self, tmpdir):
        path = os.path.join(tmpdir, 'test.fa')
        with open(path, 'w') as f:
            f.write('>seq1\nACGU\n')
        assert _detect_format(path) == 'fasta'

    def test_maf_by_extension(self, tmpdir):
        path = os.path.join(tmpdir, 'test.maf')
        with open(path, 'w') as f:
            f.write('##maf version=1\n')
        assert _detect_format(path) == 'maf'

    def test_stockholm_by_content(self, tmpdir):
        path = os.path.join(tmpdir, 'test.txt')
        with open(path, 'w') as f:
            f.write('# STOCKHOLM 1.0\nseq1 ACGU\n//\n')
        assert _detect_format(path) == 'stockholm'

    def test_fasta_by_content(self, tmpdir):
        path = os.path.join(tmpdir, 'test.txt')
        with open(path, 'w') as f:
            f.write('>seq1\nACGU\n')
        assert _detect_format(path) == 'fasta'


# ── MSA loading tests ───────────────────────────────────────────────────────

class TestLoadMsa:
    def test_load_stockholm(self, rf00390_path):
        msa = load_msa(rf00390_path)
        assert msa['alignment'].ndim == 2
        assert len(msa['leaf_names']) > 0
        assert msa['ss_cons'] is not None
        assert msa['reference_name'] == msa['leaf_names'][0]
        assert len(msa['reference_coords']) == msa['alignment'].shape[1]

    def test_load_fasta(self, fasta_msa):
        msa = load_msa(fasta_msa)
        assert msa['alignment'].shape == (3, 12)
        assert msa['leaf_names'] == ['seq1', 'seq2', 'seq3']
        assert msa['ss_cons'] is None

    def test_load_maf(self, maf_msa):
        msa = load_msa(maf_msa)
        assert msa['alignment'].shape[0] == 3
        assert len(msa['leaf_names']) == 3
        assert 'maf_metadata' in msa

    def test_reference_coords(self, fasta_msa):
        msa = load_msa(fasta_msa)
        coords = msa['reference_coords']
        # seq1 has no gaps, so coords should be 0..11
        expected = np.arange(12)
        np.testing.assert_array_equal(coords, expected)

    def test_reference_coords_with_gaps(self, tmpdir):
        path = os.path.join(tmpdir, 'gapped.fa')
        with open(path, 'w') as f:
            f.write('>ref\nAC-GU\n')
            f.write('>seq2\nACGGU\n')
        msa = load_msa(path)
        coords = msa['reference_coords']
        # ref has gap at position 2
        assert coords[0] == 0
        assert coords[1] == 1
        assert coords[2] == -1  # gap in reference
        assert coords[3] == 2
        assert coords[4] == 3

    def test_custom_reference(self, fasta_msa):
        msa = load_msa(fasta_msa, reference='seq2')
        assert msa['reference_name'] == 'seq2'

    def test_token_map(self):
        tok_map = _build_token_map('ACGU')
        assert tok_map['A'] == 0
        assert tok_map['a'] == 0
        assert tok_map['T'] == 3  # T maps to U
        assert tok_map['t'] == 3


# ── Tree building tests ─────────────────────────────────────────────────────

class TestBuildTree:
    def test_build_nj_tree(self, fasta_msa):
        msa = load_msa(fasta_msa)
        tree, nj_result, full_alignment = build_tree(
            msa['alignment'], msa['leaf_names'])
        R = len(nj_result['parentIndex'])
        assert R >= 3  # At least 3 leaves
        assert full_alignment.shape[0] == R
        assert full_alignment.shape[1] == msa['alignment'].shape[1]
        # tree may be None if subby.jax is not installed
        assert nj_result is not None


# ── Grammar selection tests ──────────────────────────────────────────────────

class TestGetGrammarForAnalysis:
    def test_secondary_structure(self):
        grammar, xg = get_grammar_for_analysis('secondary_structure')
        assert grammar is not None
        assert xg is None
        assert len(grammar.nonterminals) > 0

    def test_coding(self):
        grammar, xg = get_grammar_for_analysis('coding')
        assert grammar is not None
        assert any('E' in nt.name or 'IG' in nt.name
                    for nt in grammar.nonterminals)

    def test_pseudoknot(self):
        grammar, xg = get_grammar_for_analysis('pseudoknot', max_span=10)
        assert grammar is not None
        assert any(nt.fan_out == 2 for nt in grammar.nonterminals)

    def test_conservation(self):
        grammar, xg = get_grammar_for_analysis('conservation')
        assert grammar is not None
        assert len(grammar.rules) > 0

    def test_custom_grammar_file(self, data_dir):
        pfold_path = os.path.join(data_dir, 'pfold.eg')
        grammar, xg = get_grammar_for_analysis('custom',
                                                grammar_file=pfold_path)
        assert grammar is not None
        assert xg is not None

    def test_invalid_type(self):
        with pytest.raises(ValueError):
            get_grammar_for_analysis('nonexistent')


# ── Terminal weight computation tests ────────────────────────────────────────

class TestComputeTerminalWeights:
    def test_uniform_weights(self):
        grammar, _ = get_grammar_for_analysis('secondary_structure')
        C = 10
        alignment = np.zeros((5, C), dtype=np.int32)
        tw = compute_terminal_weights(alignment, None, grammar, C=C)
        assert tw.C == C
        assert tw.single.shape[1] == C

    def test_uniform_weights_paired_grammar(self):
        grammar, _ = get_grammar_for_analysis('secondary_structure')
        C = 8
        alignment = np.zeros((5, C), dtype=np.int32)
        tw = compute_terminal_weights(alignment, None, grammar, C=C)
        # pfold needs paired weights
        assert tw.paired is not None
        assert tw.paired.shape[1] == C
        assert tw.paired.shape[2] == C


# ── End-to-end pipeline tests ───────────────────────────────────────────────

class TestMsaToJbrowse:
    def test_fasta_secondary_structure(self, fasta_msa, tmpdir):
        result = msa_to_jbrowse(
            fasta_msa, tmpdir,
            analysis_types=['secondary_structure'],
            chrom='chr1', prefix='test')

        assert 'tracks' in result
        assert 'labels' in result
        assert 'log_probs' in result
        assert 'features' in result

        assert 'secondary_structure' in result['labels']
        assert 'secondary_structure' in result['log_probs']
        assert 'secondary_structure' in result['features']

        # Check files were created
        for path in result['tracks'].values():
            assert os.path.exists(path)
            assert os.path.getsize(path) > 0

    def test_fasta_coding(self, fasta_msa, tmpdir):
        result = msa_to_jbrowse(
            fasta_msa, tmpdir,
            analysis_types=['coding'],
            chrom='chr1', prefix='test')

        assert 'coding' in result['labels']

    def test_fasta_conservation(self, fasta_msa, tmpdir):
        result = msa_to_jbrowse(
            fasta_msa, tmpdir,
            analysis_types=['conservation'],
            chrom='chr1', prefix='test')

        assert 'conservation' in result['labels']

    def test_multiple_analyses(self, fasta_msa, tmpdir):
        result = msa_to_jbrowse(
            fasta_msa, tmpdir,
            analysis_types=['secondary_structure', 'coding'],
            chrom='chr1', prefix='test')

        assert 'secondary_structure' in result['labels']
        assert 'coding' in result['labels']

    def test_stockholm_input(self, rf00390_path, tmpdir):
        result = msa_to_jbrowse(
            rf00390_path, tmpdir,
            analysis_types=['secondary_structure'],
            prefix='rf00390')

        assert 'secondary_structure' in result['labels']
        # Check features have correct column count
        labels = result['labels']['secondary_structure']
        assert len(labels) > 0

    def test_custom_coordinates(self, fasta_msa, tmpdir):
        result = msa_to_jbrowse(
            fasta_msa, tmpdir,
            analysis_types=['conservation'],
            chrom='chrX', start=50000, strand='+',
            prefix='test')

        features = result['features']['conservation']
        for f in features:
            assert f['chrom'] == 'chrX'
            assert f['start'] >= 50000
            assert f['strand'] == '+'

    def test_output_files_structure(self, fasta_msa, tmpdir):
        result = msa_to_jbrowse(
            fasta_msa, tmpdir,
            analysis_types=['secondary_structure'],
            prefix='mytrack')

        # Should have BED, BED12, GFF3, and conservation files
        tracks = result['tracks']
        bed_files = [p for p in tracks.values() if p.endswith('.bed')]
        gff_files = [p for p in tracks.values() if p.endswith('.gff3')]
        bg_files = [p for p in tracks.values() if p.endswith('.bedGraph')]
        assert len(bed_files) >= 1
        assert len(gff_files) >= 1
        assert len(bg_files) >= 1

    def test_maf_input(self, maf_msa, tmpdir):
        result = msa_to_jbrowse(
            maf_msa, tmpdir,
            analysis_types=['conservation'],
            prefix='maf_test')

        assert 'conservation' in result['labels']
