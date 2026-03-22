"""Annotate poliovirus with XDecoder: RNA structure in coding regions.

Reproduces the analysis from Westesson & Holmes (2012) "Developing and
Applying Heterogeneous Phylogenetic Models with XRate", PLOS ONE 7(6):e36898.

XDecoder detects conserved RNA secondary structures overlapping protein-coding
regions by combining:
  - Codon-position-specific substitution rates (HKY with per-position kappa)
  - 4 rate classes for non-structural evolution
  - Paired substitution models for base-pair co-evolution
  - An SCFG grammar switching between structural and non-structural regions

Pipeline:
  1. Load XDecoder.eg grammar (macro-expanded from xrate format)
  2. Fetch poliovirus sequences from NCBI and align with MAFFT
  3. Build NJ tree from alignment distances
  4. Compute phylogenetic terminal weights via subby
  5. Run inside-outside for posterior structure probabilities
  6. Output wiggle (.wig) tracks showing folding potential

Known poliovirus RNA structures (for validation):
  - 5' cloverleaf (nt 1-88): essential for replication
  - IRES (nt 124-745): internal ribosome entry site, domains II-VI
  - cre element (nt 4444-4505): cis-acting replication element in 2C
  - 3' UTR structures (nt ~7370-7440)

Usage:
    python examples/poliovirus_xdecoder.py [--region START:END] [--output DIR]

Requires: subby, biopython (for NCBI fetch), mafft (for alignment)
"""

import argparse
import os
import sys
import subprocess
import tempfile

import numpy as np
import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

from jaxrate import (
    compile_grammar, viterbi, TerminalWeights,
    neighbor_joining, to_subby_tree, jukes_cantor_distances,
)
from jaxrate.presets import xdecoder_grammar
from jaxrate.scfg import scfg_inside, scfg_outside, scfg_posteriors
from jaxrate.grammar import classify_grammar, validate_grammar
from jaxrate.wiggle import (
    write_wiggle, write_multi_wiggle, posteriors_to_structure_track,
)


# Poliovirus 1 Mahoney reference accession
REFERENCE_ACCESSION = 'V01149'

# Representative enterovirus accessions for the alignment
# (subset of sequences used in Westesson & Holmes 2012)
ACCESSIONS = [
    'V01149',   # Poliovirus 1 Mahoney
    'AY184219',  # Poliovirus 1 Sabin
    'AY184220',  # Poliovirus 2 Sabin
    'AY184221',  # Poliovirus 3 Sabin
    'X00595',   # Poliovirus 2 Lansing
    'K01392',   # Poliovirus 3 Leon
    'D00627',   # Coxsackievirus A21
    'M16560',   # Coxsackievirus B3
    'D00538',   # Human rhinovirus 14
    'X01087',   # Human rhinovirus 2
]

# Known RNA structure regions in poliovirus 1 (1-based coordinates)
KNOWN_STRUCTURES = {
    "5' cloverleaf": (1, 88),
    "IRES domain II": (124, 188),
    "IRES domain III": (189, 468),
    "IRES domain IV": (469, 620),
    "IRES domain V": (621, 726),
    "IRES domain VI": (727, 745),
    "cre (2C)": (4444, 4505),
    "3' UTR": (7370, 7440),
}


def fetch_sequences(accessions, output_fasta):
    """Fetch sequences from NCBI GenBank and write to FASTA."""
    try:
        from Bio import Entrez, SeqIO
    except ImportError:
        print("ERROR: BioPython required for NCBI fetch. Install: pip install biopython")
        print("Alternatively, provide a pre-aligned FASTA file with --alignment")
        sys.exit(1)

    Entrez.email = "jaxrate@example.com"
    print(f"Fetching {len(accessions)} sequences from NCBI...")

    with open(output_fasta, 'w') as out:
        for acc in accessions:
            handle = Entrez.efetch(db="nucleotide", id=acc,
                                   rettype="fasta", retmode="text")
            record = SeqIO.read(handle, "fasta")
            handle.close()
            # Use accession as ID for cleaner names
            record.id = acc
            record.description = ''
            SeqIO.write(record, out, "fasta")
            print(f"  {acc}: {len(record.seq)} nt")

    return output_fasta


def align_sequences(fasta_path, aligned_path):
    """Align sequences with MAFFT."""
    print("Aligning with MAFFT...")
    try:
        result = subprocess.run(
            ['mafft', '--auto', '--quiet', fasta_path],
            capture_output=True, text=True, check=True)
        with open(aligned_path, 'w') as f:
            f.write(result.stdout)
        print(f"  Aligned to {aligned_path}")
    except FileNotFoundError:
        print("ERROR: MAFFT not found. Install: conda install -c bioconda mafft")
        print("Alternatively, provide a pre-aligned FASTA file with --alignment")
        sys.exit(1)
    return aligned_path


def parse_fasta_alignment(filepath, alphabet='ACGU'):
    """Parse a FASTA alignment into a numeric array.

    Returns:
        alignment: (N, C) int32 array (0=A, 1=C, 2=G, 3=U, 99=gap)
        names: list of sequence names
    """
    names = []
    seqs = []
    current_name = None
    current_seq = []

    with open(filepath) as f:
        for line in f:
            line = line.strip()
            if line.startswith('>'):
                if current_name is not None:
                    names.append(current_name)
                    seqs.append(''.join(current_seq))
                current_name = line[1:].split()[0]
                current_seq = []
            else:
                current_seq.append(line)
    if current_name is not None:
        names.append(current_name)
        seqs.append(''.join(current_seq))

    tok_map = {c: i for i, c in enumerate(alphabet)}
    # Also map lowercase and T/t
    for c in alphabet:
        tok_map[c.lower()] = tok_map[c]
    tok_map['T'] = tok_map.get('U', tok_map.get('T', 3))
    tok_map['t'] = tok_map['T']

    C = max(len(s) for s in seqs)
    N = len(seqs)
    alignment = np.full((N, C), 99, dtype=np.int32)

    for i, seq in enumerate(seqs):
        for j, ch in enumerate(seq):
            alignment[i, j] = tok_map.get(ch, 99)

    return alignment, names


def reorder_alignment_for_tree(alignment, tree_result, leaf_names):
    """Reorder alignment rows and add gap rows for internal nodes."""
    N, C = alignment.shape
    R = len(tree_result['parentIndex'])
    gap_token = 99

    name_to_row = {name: i for i, name in enumerate(leaf_names)}
    full_alignment = np.full((R, C), gap_token, dtype=np.int32)

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


def extract_region(alignment, start, end):
    """Extract columns start:end (0-based) from alignment."""
    return alignment[:, start:end]


def compute_terminal_weights(alignment, tree, xg, C):
    """Compute terminal weights for XDecoder grammar.

    XDecoder has both single-position and paired-position chains.
    This computes phylogenetic log-likelihoods for each.
    """
    from subby.jax import LogLike
    from subby.formats import all_column_ktuples, kmer_tokenize

    subby_models = xg.to_subby_models()

    # Single-column weights: one per chain model
    single_lls = []
    for model in subby_models['single']:
        ll = LogLike(jnp.array(alignment), tree, model)
        single_lls.append(ll)
    single = jnp.stack(single_lls, axis=0)  # (K_single, C)
    print(f"  Single-column weights: {single.shape}")

    # Paired-column weights
    paired = None
    if subby_models['paired']:
        paired_lls = []
        for pm in subby_models['paired']:
            p_model = pm['model']
            A = pm['A']
            tuples = all_column_ktuples(C, 2, ordered=True)
            if len(tuples) > 0:
                kt = kmer_tokenize(np.asarray(alignment), A, tuples, gap_mode='any')
                ll = LogLike(jnp.array(kt['alignment']), tree, p_model)
                ll_matrix = jnp.full((C, C), -1e38)
                for t_idx in range(len(tuples)):
                    ci, cj = tuples[t_idx]
                    ll_matrix = ll_matrix.at[ci, cj].set(ll[t_idx])
                paired_lls.append(ll_matrix)
            else:
                paired_lls.append(jnp.zeros((C, C)))
        paired = jnp.stack(paired_lls, axis=0)
        print(f"  Paired-column weights: {paired.shape}")

    return TerminalWeights(single=single, paired=paired, C=C)


def synthetic_terminal_weights(xg, C, seed=42):
    """Generate synthetic terminal weights for testing without subby.

    Creates plausible log-likelihood values: structural regions have
    higher paired emission scores.
    """
    rng = np.random.RandomState(seed)

    n_single = len(xg.get_single_models())
    n_paired = len(xg.get_paired_models())

    # Single-column: baseline log-likelihoods ~ -2.0 ± 0.5
    single = jnp.array(rng.randn(n_single, C) * 0.5 - 2.0)

    # Paired-column: mostly very negative (no pairing),
    # with some elevated regions at known structure positions
    paired = jnp.array(rng.randn(n_paired, C, C) * 0.3 - 4.0)

    return TerminalWeights(single=single, paired=paired, C=C)


def main():
    parser = argparse.ArgumentParser(
        description='Annotate poliovirus with XDecoder RNA structure grammar')
    parser.add_argument('--alignment', type=str, default=None,
                        help='Pre-aligned FASTA file (skip fetch+align)')
    parser.add_argument('--region', type=str, default=None,
                        help='Region to analyze, e.g. "1:745" (1-based)')
    parser.add_argument('--output', type=str, default='.',
                        help='Output directory for wiggle files')
    parser.add_argument('--synthetic', action='store_true',
                        help='Use synthetic terminal weights (no subby needed)')
    parser.add_argument('--viterbi-only', action='store_true',
                        help='Run Viterbi only (skip posteriors)')
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)

    # --- Step 1: Load XDecoder grammar ---
    print("=" * 60)
    print("Step 1: Load XDecoder grammar")
    print("=" * 60)
    xg = xdecoder_grammar()
    grammar = xg.grammar
    print(f"  Nonterminals: {len(grammar.nonterminals)}")
    print(f"  Rules: {len(grammar.rules)}")
    print(f"  Single chains: {len(xg.get_single_models())}")
    print(f"  Paired chains: {len(xg.get_paired_models())}")
    print(f"  Classification: {classify_grammar(grammar)}")
    validate_grammar(grammar)
    print("  Validation: OK")

    # --- Step 2: Get alignment ---
    print("\n" + "=" * 60)
    print("Step 2: Get poliovirus alignment")
    print("=" * 60)

    if args.alignment:
        aligned_path = args.alignment
        print(f"  Using provided alignment: {aligned_path}")
    else:
        tmpdir = tempfile.mkdtemp(prefix='xdecoder_')
        fasta_path = os.path.join(tmpdir, 'poliovirus.fasta')
        aligned_path = os.path.join(tmpdir, 'poliovirus_aligned.fasta')
        fetch_sequences(ACCESSIONS, fasta_path)
        align_sequences(fasta_path, aligned_path)

    alignment, leaf_names = parse_fasta_alignment(aligned_path)
    N, C_full = alignment.shape
    print(f"  Sequences: {N}, Alignment columns: {C_full}")
    print(f"  Leaves: {leaf_names}")

    # Extract region if specified
    region_start, region_end = 0, C_full
    if args.region:
        parts = args.region.split(':')
        region_start = int(parts[0]) - 1  # convert to 0-based
        region_end = int(parts[1]) if len(parts) > 1 else C_full
        alignment = extract_region(alignment, region_start, region_end)
        C = alignment.shape[1]
        print(f"  Analyzing region {region_start + 1}:{region_end} ({C} columns)")
    else:
        C = C_full
        print(f"  Analyzing full alignment ({C} columns)")
        if C > 500:
            print(f"  WARNING: SCFG algorithms are O(C^3 * K^3).")
            print(f"  With C={C}, K={len(grammar.nonterminals)}, this will be slow.")
            print(f"  Consider using --region to analyze a sub-region,")
            print(f"  e.g. --region 1:200 for the 5' cloverleaf + IRES domain II")

    # --- Step 3: Build NJ tree ---
    print("\n" + "=" * 60)
    print("Step 3: Build NJ tree")
    print("=" * 60)
    D = jukes_cantor_distances(alignment, A=4)
    nj_result = neighbor_joining(D, leaf_names)
    tree = to_subby_tree(nj_result)
    R = len(nj_result['parentIndex'])
    print(f"  Tree nodes: {R} ({N} leaves + {R - N} internal)")
    if nj_result.get('newick'):
        print(f"  Newick: {nj_result['newick'][:80]}...")

    full_alignment = reorder_alignment_for_tree(alignment, nj_result, leaf_names)
    print(f"  Full alignment shape: {full_alignment.shape}")

    # --- Step 4: Terminal weights ---
    print("\n" + "=" * 60)
    print("Step 4: Compute terminal weights")
    print("=" * 60)
    if args.synthetic:
        print("  Using synthetic terminal weights")
        tw = synthetic_terminal_weights(xg, C)
    else:
        tw = compute_terminal_weights(full_alignment, tree, xg, C)

    # --- Step 5: Run algorithms ---
    print("\n" + "=" * 60)
    print("Step 5: Run grammar algorithms")
    print("=" * 60)

    # Viterbi decoding
    print("  Running Viterbi decoding...")
    labels, log_prob = viterbi(grammar, tw)
    print(f"  Viterbi log-probability: {float(log_prob):.4f}")

    nt_names = [nt.name for nt in grammar.nonterminals]

    # Map labels to structure annotation
    struct_labels = []
    for c in range(C):
        name = nt_names[int(labels[c])]
        if 'pfoldCodingF' in name and '*' not in name:
            struct_labels.append('<')  # paired/stem
        elif 'pfoldCodingU' in name and '*' not in name:
            struct_labels.append('.')  # loop
        elif 'ns' in name and 'emit' in name and '*' not in name:
            struct_labels.append(',')  # non-structural
        else:
            struct_labels.append('_')  # null/transition

    viterbi_annotation = ''.join(struct_labels)

    # Posteriors (inside-outside)
    if not args.viterbi_only:
        print("  Running inside-outside for posteriors...")
        cg = compile_grammar(grammar)
        posteriors, log_ll = scfg_posteriors(cg, tw)
        print(f"  Inside log-likelihood: {float(log_ll):.4f}")

        # Structure probability track
        struct_prob = posteriors_to_structure_track(posteriors, grammar)
        print(f"  Structure probability range: [{struct_prob.min():.4f}, {struct_prob.max():.4f}]")
    else:
        posteriors = None
        struct_prob = None

    # --- Step 6: Output ---
    print("\n" + "=" * 60)
    print("Step 6: Write output files")
    print("=" * 60)

    chrom = 'Poliovirus1'
    wig_start = region_start + 1  # 1-based for WIG

    # Viterbi structure annotation (binary: 1=structural, 0=non-structural)
    viterbi_struct = np.array([1.0 if c in '<.' else 0.0 for c in struct_labels])
    viterbi_wig = os.path.join(args.output, 'xdecoder_viterbi.wig')
    write_wiggle(viterbi_wig, viterbi_struct,
                 chrom=chrom, start=wig_start,
                 track_name='XDecoder_Viterbi',
                 description='XDecoder Viterbi structure annotation')
    print(f"  Viterbi track: {viterbi_wig}")

    if struct_prob is not None:
        # Posterior structure probability
        posterior_wig = os.path.join(args.output, 'xdecoder_posterior.wig')
        write_wiggle(posterior_wig, struct_prob,
                     chrom=chrom, start=wig_start,
                     track_name='XDecoder_Posterior',
                     description='XDecoder posterior structure probability')
        print(f"  Posterior track: {posterior_wig}")

        # Multi-track file with both
        multi_wig = os.path.join(args.output, 'xdecoder_tracks.wig')
        write_multi_wiggle(multi_wig, [
            {'name': 'XDecoder_Viterbi',
             'description': 'Viterbi structure (0/1)',
             'values': viterbi_struct},
            {'name': 'XDecoder_FoldPotential',
             'description': 'Posterior P(structure)',
             'values': struct_prob},
        ], chrom=chrom, start=wig_start)
        print(f"  Combined tracks: {multi_wig}")

    # --- Step 7: Validation against known structures ---
    print("\n" + "=" * 60)
    print("Step 7: Validate against known RNA structures")
    print("=" * 60)

    for name, (s, e) in KNOWN_STRUCTURES.items():
        # Adjust to current region
        s_adj = s - 1 - region_start  # 0-based, relative to region
        e_adj = e - region_start
        if s_adj < 0 or e_adj > C:
            print(f"  {name} ({s}-{e}): outside analyzed region")
            continue

        # Viterbi: fraction of positions annotated as structural
        region_labels = viterbi_struct[s_adj:e_adj]
        vit_frac = region_labels.mean()

        msg = f"  {name} ({s}-{e}): Viterbi {vit_frac:.1%} structural"

        if struct_prob is not None:
            post_mean = struct_prob[s_adj:e_adj].mean()
            msg += f", Posterior mean {post_mean:.3f}"

        print(msg)

    # Print annotation for first 100 columns
    print("\n" + "=" * 60)
    print("Annotation preview (first 100 columns)")
    print("=" * 60)
    show_len = min(100, C)
    # Reference sequence
    ref_seq = []
    for c in range(show_len):
        tok = alignment[0, c]
        ref_seq.append('ACGU'[tok] if tok < 4 else '-')
    print(f"  Sequence:   {''.join(ref_seq)}")
    print(f"  Structure:  {viterbi_annotation[:show_len]}")
    if struct_prob is not None:
        # Discretize posterior to characters
        post_chars = []
        for v in struct_prob[:show_len]:
            if v > 0.8:
                post_chars.append('#')
            elif v > 0.5:
                post_chars.append('*')
            elif v > 0.2:
                post_chars.append('+')
            elif v > 0.05:
                post_chars.append('.')
            else:
                post_chars.append(' ')
        print(f"  Post. P(s): {''.join(post_chars)}")

    print("\nDone.")


if __name__ == '__main__':
    main()
