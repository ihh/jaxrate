"""High-level pipeline: MSA → grammar analysis → JBrowse tracks.

Converts genome-aligned MSAs (with or without trees) into JBrowse-browsable
annotation tracks using various grammars for secondary structure, coding
structure, conservation, and motif enrichment.

Supported input formats: Stockholm (.sto), FASTA (.fa/.fasta), MAF (.maf)
Supported output: BED6, BED12, GFF3, BedGraph, WIG
"""

import os
import numpy as np

from .types import TerminalWeights
from .grammar import classify_grammar, compile_grammar
from .viterbi import viterbi
from .jbrowse import (
    labels_to_features,
    merge_features,
    write_bed,
    write_gff3,
    conservation_bedgraph,
    posteriors_to_bedgraph,
    write_jbrowse_tracks,
)


# ── MSA loading ──────────────────────────────────────────────────────────────

def load_msa(filepath, format=None, alphabet='ACGU', reference=None):
    """Load a multiple sequence alignment from file.

    Args:
        filepath: path to MSA file
        format: 'stockholm', 'fasta', or 'maf'. Auto-detected if None.
        alphabet: token alphabet (default RNA)
        reference: name of reference sequence for coordinate mapping.
            If None, uses the first sequence.

    Returns:
        dict with keys:
            alignment: (N, C) int32 tokenized alignment
            leaf_names: list of sequence names
            ss_cons: SS_cons annotation string (Stockholm only, else None)
            reference_name: name of reference sequence
            reference_coords: (C,) int array mapping alignment columns to
                reference positions (-1 for gaps in reference)
    """
    if format is None:
        format = _detect_format(filepath)

    if format == 'stockholm':
        return _load_stockholm(filepath, alphabet, reference)
    elif format == 'fasta':
        return _load_fasta(filepath, alphabet, reference)
    elif format == 'maf':
        return _load_maf(filepath, alphabet, reference)
    else:
        raise ValueError(f"Unknown format: {format}. Use 'stockholm', 'fasta', or 'maf'.")


def _detect_format(filepath):
    """Auto-detect MSA format from file extension or content."""
    ext = os.path.splitext(filepath)[1].lower()
    if ext in ('.sto', '.stk', '.stockholm'):
        return 'stockholm'
    elif ext in ('.fa', '.fasta', '.fna', '.fas'):
        return 'fasta'
    elif ext == '.maf':
        return 'maf'

    # Peek at content
    with open(filepath) as f:
        first_line = f.readline().strip()
    if first_line.startswith('# STOCKHOLM'):
        return 'stockholm'
    elif first_line.startswith('>'):
        return 'fasta'
    elif first_line.startswith('##maf'):
        return 'maf'

    raise ValueError(f"Cannot detect format of {filepath}. "
                     "Specify format='stockholm', 'fasta', or 'maf'.")


def _load_stockholm(filepath, alphabet, reference):
    """Load Stockholm-format MSA."""
    tok_map = _build_token_map(alphabet)

    names = []
    seqs = {}
    ss_cons = None

    with open(filepath) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#=GF') or line == '//':
                continue
            if line.startswith('# STOCKHOLM'):
                continue
            if line.startswith('#=GC SS_cons'):
                ss_cons = line.split()[-1]
                continue
            if line.startswith('#'):
                continue
            parts = line.split()
            if len(parts) >= 2:
                name, seq = parts[0], parts[-1]
                if name not in seqs:
                    names.append(name)
                    seqs[name] = []
                seqs[name].append(seq)

    # Concatenate multi-block sequences
    sequences = {name: ''.join(blocks) for name, blocks in seqs.items()}
    return _finalize_msa(names, sequences, tok_map, reference, ss_cons)


def _load_fasta(filepath, alphabet, reference):
    """Load FASTA-format MSA."""
    tok_map = _build_token_map(alphabet)
    names = []
    sequences = {}
    current = None

    with open(filepath) as f:
        for line in f:
            line = line.strip()
            if line.startswith('>'):
                current = line[1:].split()[0]
                names.append(current)
                sequences[current] = []
            elif current is not None:
                sequences[current].append(line)

    sequences = {n: ''.join(s) for n, s in sequences.items()}
    return _finalize_msa(names, sequences, tok_map, reference, None)


def _load_maf(filepath, alphabet, reference):
    """Load MAF (Multiple Alignment Format) file.

    MAF format has alignment blocks with 's' lines:
    s species.chrom start size strand srcSize sequence
    """
    tok_map = _build_token_map(alphabet)
    names = []
    sequences = {}
    maf_metadata = {}

    with open(filepath) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            if line.startswith('s '):
                parts = line.split()
                # s src start size strand srcSize sequence
                src = parts[1]
                maf_start = int(parts[2])
                maf_size = int(parts[3])
                maf_strand = parts[4]
                maf_src_size = int(parts[5])
                seq = parts[6]

                name = src
                if name not in sequences:
                    names.append(name)
                    sequences[name] = []
                    maf_metadata[name] = {
                        'start': maf_start,
                        'size': maf_size,
                        'strand': maf_strand,
                        'src_size': maf_src_size,
                    }
                sequences[name].append(seq)

    sequences = {n: ''.join(s) for n, s in sequences.items()}
    result = _finalize_msa(names, sequences, tok_map, reference, None)
    result['maf_metadata'] = maf_metadata
    return result


def _build_token_map(alphabet):
    """Build character -> token index map."""
    tok_map = {}
    for i, c in enumerate(alphabet):
        tok_map[c] = i
        tok_map[c.lower()] = i
    # Handle T/U equivalence
    if 'U' in alphabet:
        tok_map['T'] = tok_map['U']
        tok_map['t'] = tok_map['U']
    elif 'T' in alphabet:
        tok_map['U'] = tok_map['T']
        tok_map['u'] = tok_map['T']
    return tok_map


def _finalize_msa(names, sequences, tok_map, reference, ss_cons):
    """Common finalization: tokenize and compute reference coordinates."""
    if not names:
        raise ValueError("No sequences found in alignment")

    C = max(len(s) for s in sequences.values())
    N = len(names)
    gap_token = 99

    alignment = np.full((N, C), gap_token, dtype=np.int32)
    for i, name in enumerate(names):
        seq = sequences[name]
        for j, ch in enumerate(seq):
            alignment[i, j] = tok_map.get(ch, gap_token)

    ref_name = reference if reference else names[0]
    ref_idx = names.index(ref_name) if ref_name in names else 0
    ref_name = names[ref_idx]

    # Map alignment columns to reference positions
    ref_coords = np.full(C, -1, dtype=np.int64)
    ref_pos = 0
    for j in range(C):
        if alignment[ref_idx, j] != gap_token:
            ref_coords[j] = ref_pos
            ref_pos += 1

    return {
        'alignment': alignment,
        'leaf_names': names,
        'ss_cons': ss_cons,
        'reference_name': ref_name,
        'reference_coords': ref_coords,
    }


# ── Tree construction ────────────────────────────────────────────────────────

def build_tree(alignment, leaf_names, method='nj'):
    """Build a phylogenetic tree from alignment distances.

    Args:
        alignment: (N, C) int32 tokenized alignment
        leaf_names: list of N sequence names
        method: 'nj' for Neighbor-Joining (only option currently)

    Returns:
        tree: subby Tree NamedTuple (or None if subby.jax unavailable)
        nj_result: dict with parentIndex, distanceToParent, etc.
        full_alignment: (R, C) int32 with rows for all tree nodes
    """
    from .nj import neighbor_joining, jukes_cantor_distances

    D = jukes_cantor_distances(alignment, A=4)
    nj_result = neighbor_joining(D, leaf_names)

    try:
        from .nj import to_subby_tree
        tree = to_subby_tree(nj_result)
    except ImportError:
        tree = None

    # Reorder alignment for tree node ordering
    N, C = alignment.shape
    R = len(nj_result['parentIndex'])
    gap_token = 99

    name_to_row = {name: i for i, name in enumerate(leaf_names)}
    full_alignment = np.full((R, C), gap_token, dtype=np.int32)

    children_count = np.zeros(R, dtype=np.int32)
    for i in range(1, R):
        children_count[nj_result['parentIndex'][i]] += 1
    is_leaf = children_count == 0

    tree_leaf_pos = 0
    for node_idx in range(R):
        if is_leaf[node_idx]:
            leaf_name = nj_result['leaf_names'][tree_leaf_pos]
            if leaf_name in name_to_row:
                full_alignment[node_idx] = alignment[name_to_row[leaf_name]]
            tree_leaf_pos += 1

    return tree, nj_result, full_alignment


# ── Grammar selection ────────────────────────────────────────────────────────

def get_grammar_for_analysis(analysis_type, grammar_file=None, **kwargs):
    """Get a grammar for a specific analysis type.

    Args:
        analysis_type: one of 'secondary_structure', 'coding', 'pseudoknot',
            'conservation', 'xdecoder', 'custom'
        grammar_file: path to .eg grammar file (for 'custom' or override)
        **kwargs: extra args passed to grammar constructors

    Returns:
        grammar: Grammar
        xrate_grammar: XrateGrammar (if parsed from file, else None)
    """
    if grammar_file:
        from .xrate_parser import parse_xrate_file
        xg = parse_xrate_file(grammar_file)
        return xg.grammar, xg

    from .presets import (
        pfold_grammar, gene_finder_grammar, pseudoknot_grammar,
        xdecoder_grammar,
    )

    if analysis_type == 'secondary_structure':
        return pfold_grammar(), None
    elif analysis_type == 'coding':
        return gene_finder_grammar(), None
    elif analysis_type == 'pseudoknot':
        max_span = kwargs.get('max_span', None)
        return pseudoknot_grammar(max_span=max_span), None
    elif analysis_type == 'xdecoder':
        xg = xdecoder_grammar()
        return xg.grammar, xg
    elif analysis_type == 'conservation':
        # For conservation, use a simple null model
        # that just emits single columns
        from .grammar import GrammarBuilder
        from .types import EmissionGroup
        import math
        gb = GrammarBuilder()
        S = gb.add_nonterminal('conserved')
        gb.add_rule(S, rhs=[S], emissions=[EmissionGroup(1, 0)],
                    log_weight=math.log(0.99))
        gb.add_rule(S, rhs=[], emissions=[EmissionGroup(1, 0)],
                    log_weight=math.log(0.01))
        return gb.build(start=S, n_models=1), None
    else:
        raise ValueError(f"Unknown analysis type: {analysis_type}. "
                         "Use 'secondary_structure', 'coding', 'pseudoknot', "
                         "'xdecoder', 'conservation', or 'custom'.")


# ── Terminal weight computation ──────────────────────────────────────────────

def compute_terminal_weights(full_alignment, tree, grammar, xrate_grammar=None,
                             C=None):
    """Compute terminal weights, choosing method based on available models.

    If xrate_grammar is provided and subby.jax is available, extracts subby
    models from it. Otherwise, creates synthetic uniform weights.

    Args:
        full_alignment: (R, C) int32 alignment (with tree node rows)
        tree: subby Tree (or None if subby.jax unavailable)
        grammar: Grammar
        xrate_grammar: optional XrateGrammar with chain definitions
        C: number of columns (inferred from alignment if None)

    Returns:
        TerminalWeights
    """
    if C is None:
        C = full_alignment.shape[1]

    if xrate_grammar is not None and tree is not None:
        try:
            return _compute_phylo_weights(full_alignment, tree,
                                          xrate_grammar, C)
        except ImportError:
            pass

    return _compute_uniform_weights(grammar, C)


def _compute_phylo_weights(full_alignment, tree, xrate_grammar, C):
    """Compute phylogenetic terminal weights using subby."""
    import jax.numpy as jnp
    from subby.jax import LogLike
    from subby.formats import all_column_ktuples, kmer_tokenize

    subby_models = xrate_grammar.to_subby_models()

    # Single-column
    single_lls = []
    for model in subby_models['single']:
        ll = LogLike(jnp.array(full_alignment), tree, model)
        single_lls.append(ll)
    single = jnp.stack(single_lls, axis=0)

    # Paired-column
    paired = None
    if subby_models.get('paired'):
        paired_lls = []
        for pm in subby_models['paired']:
            p_model = pm['model']
            A = pm['A']
            tuples = all_column_ktuples(C, 2, ordered=True)
            if len(tuples) > 0:
                kt = kmer_tokenize(np.asarray(full_alignment), A, tuples,
                                   gap_mode='any')
                ll = LogLike(jnp.array(kt['alignment']), tree, p_model)
                ll_matrix = jnp.full((C, C), -1e38)
                for t_idx in range(len(tuples)):
                    ci, cj = tuples[t_idx]
                    ll_matrix = ll_matrix.at[ci, cj].set(ll[t_idx])
                paired_lls.append(ll_matrix)
            else:
                paired_lls.append(jnp.zeros((C, C)))
        paired = jnp.stack(paired_lls, axis=0)

    return TerminalWeights(single=single, paired=paired, C=C)


def _compute_uniform_weights(grammar, C):
    """Create uniform terminal weights when no substitution models available."""
    import jax.numpy as jnp

    single = jnp.zeros((grammar.n_models, C))

    # Check if grammar needs paired weights
    needs_paired = any(
        any(e.n_positions == 2 for e in rule.emissions)
        for rule in grammar.rules
    )
    paired = jnp.zeros((grammar.n_models, C, C)) if needs_paired else None

    return TerminalWeights(single=single, paired=paired, C=C)


# ── Main pipeline ────────────────────────────────────────────────────────────

def msa_to_jbrowse(msa_path, output_dir, analysis_types=None,
                   grammar_file=None, tree_file=None,
                   chrom=None, start=0, strand='.',
                   reference=None, prefix='jaxrate',
                   compute_posteriors=False, format=None):
    """Convert an MSA file to JBrowse annotation tracks.

    This is the main entry point for the MSA-to-JBrowse pipeline.

    Args:
        msa_path: path to MSA file (Stockholm, FASTA, or MAF)
        output_dir: output directory for track files
        analysis_types: list of analysis types to run. Default:
            ['secondary_structure']. Options: 'secondary_structure',
            'coding', 'pseudoknot', 'xdecoder', 'conservation'
        grammar_file: path to .eg grammar file (overrides analysis_type)
        tree_file: path to Newick tree file (optional; NJ built if omitted)
        chrom: chromosome name (default: inferred from MSA)
        start: 0-based genomic start coordinate
        strand: '+', '-', or '.'
        reference: reference sequence name
        prefix: output filename prefix
        compute_posteriors: if True, also compute inside-outside posteriors
        format: MSA format override ('stockholm', 'fasta', 'maf')

    Returns:
        dict with keys:
            tracks: dict mapping track_name -> file_path
            labels: dict mapping analysis_type -> (C,) int32 labels
            log_probs: dict mapping analysis_type -> float
            features: dict mapping analysis_type -> list of feature dicts
    """
    os.makedirs(output_dir, exist_ok=True)

    if analysis_types is None:
        if grammar_file:
            analysis_types = ['custom']
        else:
            analysis_types = ['secondary_structure']

    # Load MSA
    msa = load_msa(msa_path, format=format, reference=reference)
    alignment = msa['alignment']
    leaf_names = msa['leaf_names']
    N, C = alignment.shape

    if chrom is None:
        chrom = msa['reference_name']

    # Build tree
    tree, nj_result, full_alignment = build_tree(alignment, leaf_names)

    # Run each analysis
    all_tracks = {}
    all_labels = {}
    all_log_probs = {}
    all_features = {}

    for atype in analysis_types:
        atype_key = atype if atype != 'custom' else 'custom'
        aprefix = f'{prefix}_{atype_key}'

        # Get grammar
        grammar, xg = get_grammar_for_analysis(
            atype, grammar_file=grammar_file if atype == 'custom' else None)

        # Compute terminal weights
        tw = compute_terminal_weights(full_alignment, tree, grammar,
                                      xrate_grammar=xg, C=C)

        # Run Viterbi
        labels, log_prob = viterbi(grammar, tw)
        all_labels[atype_key] = labels
        all_log_probs[atype_key] = float(log_prob)

        # Compute posteriors if requested
        posteriors = None
        if compute_posteriors:
            gclass = classify_grammar(grammar)
            if gclass in ('hmm', 'scfg'):
                cg = compile_grammar(grammar)
                if gclass == 'scfg':
                    from .scfg import scfg_posteriors
                    posteriors, _ = scfg_posteriors(cg, tw)
                else:
                    from .hmm import hmm_forward, hmm_backward
                    import jax.numpy as jnp
                    alpha, log_z = hmm_forward(cg, tw)
                    beta = hmm_backward(cg, tw)
                    posteriors = np.exp(
                        np.asarray(alpha) + np.asarray(beta)
                        - float(log_z))

        # Write tracks
        tracks = write_jbrowse_tracks(
            output_dir, labels, grammar, tw,
            posteriors=posteriors,
            chrom=chrom, start=start, strand=strand,
            prefix=aprefix)

        all_tracks.update(tracks)

        # Store features
        from .jbrowse import labels_to_features
        all_features[atype_key] = labels_to_features(
            labels, grammar, chrom=chrom, start=start, strand=strand)

    return {
        'tracks': all_tracks,
        'labels': all_labels,
        'log_probs': all_log_probs,
        'features': all_features,
    }
