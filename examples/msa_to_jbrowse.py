"""Convert genome-aligned MSA to JBrowse annotation tracks.

Usage:
    # RNA secondary structure from Stockholm alignment
    python examples/msa_to_jbrowse.py jaxrate/data/RF00390.sto -o output/

    # Gene structure from FASTA alignment
    python examples/msa_to_jbrowse.py alignment.fa -a coding -o output/

    # Multiple analyses
    python examples/msa_to_jbrowse.py alignment.sto -a secondary_structure -a coding -o output/

    # Custom xrate grammar
    python examples/msa_to_jbrowse.py alignment.sto -g jaxrate/data/pfold.eg -o output/

    # With posteriors and coordinate mapping
    python examples/msa_to_jbrowse.py alignment.sto -a secondary_structure --posteriors \
        --chrom chr1 --start 1000 -o output/

Supported MSA formats: Stockholm (.sto), FASTA (.fa), MAF (.maf)

Output tracks:
    - BED6:     Per-nonterminal feature annotations
    - BED12:    Merged gene/structure features with blocks
    - GFF3:     Hierarchical feature annotations
    - BedGraph: Conservation scores (phylogenetic log-likelihoods)
    - BedGraph: Posterior probabilities (with --posteriors)
"""

import argparse
import os
import sys

import jax
jax.config.update("jax_enable_x64", True)


def main():
    parser = argparse.ArgumentParser(
        description='Convert genome-aligned MSA to JBrowse annotation tracks',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__)

    parser.add_argument('msa', help='Input MSA file (Stockholm, FASTA, or MAF)')
    parser.add_argument('-o', '--output', default='jbrowse_output',
                        help='Output directory (default: jbrowse_output)')
    parser.add_argument('-a', '--analysis', action='append', dest='analyses',
                        choices=['secondary_structure', 'coding', 'pseudoknot',
                                 'xdecoder', 'conservation'],
                        help='Analysis type (can be repeated for multiple)')
    parser.add_argument('-g', '--grammar', default=None,
                        help='Custom xrate .eg grammar file')
    parser.add_argument('--chrom', default=None,
                        help='Chromosome name (default: reference sequence name)')
    parser.add_argument('--start', type=int, default=0,
                        help='0-based genomic start coordinate')
    parser.add_argument('--strand', default='.', choices=['+', '-', '.'],
                        help='Strand')
    parser.add_argument('--reference', default=None,
                        help='Reference sequence name for coordinate mapping')
    parser.add_argument('--prefix', default='jaxrate',
                        help='Output filename prefix')
    parser.add_argument('--posteriors', action='store_true',
                        help='Compute posterior probabilities (slower)')
    parser.add_argument('--format', default=None,
                        choices=['stockholm', 'fasta', 'maf'],
                        help='Force MSA format (default: auto-detect)')

    args = parser.parse_args()

    if not os.path.exists(args.msa):
        print(f"Error: MSA file not found: {args.msa}", file=sys.stderr)
        sys.exit(1)

    analysis_types = args.analyses
    if analysis_types is None and args.grammar is None:
        analysis_types = ['secondary_structure']

    print(f"Input MSA:  {args.msa}")
    print(f"Output dir: {args.output}")
    print(f"Analyses:   {analysis_types or ['custom']}")
    if args.grammar:
        print(f"Grammar:    {args.grammar}")
    print()

    from jaxrate.msa_to_jbrowse import msa_to_jbrowse

    result = msa_to_jbrowse(
        msa_path=args.msa,
        output_dir=args.output,
        analysis_types=analysis_types,
        grammar_file=args.grammar,
        chrom=args.chrom,
        start=args.start,
        strand=args.strand,
        reference=args.reference,
        prefix=args.prefix,
        compute_posteriors=args.posteriors,
        format=args.format,
    )

    # Summary
    print("=" * 60)
    print("Results")
    print("=" * 60)

    for atype, log_prob in result['log_probs'].items():
        print(f"\n  {atype}:")
        print(f"    Viterbi log-probability: {log_prob:.4f}")

        features = result['features'][atype]
        # Count feature types
        type_counts = {}
        for f in features:
            ft = f['feature_type']
            type_counts[ft] = type_counts.get(ft, 0) + 1
        for ft, count in sorted(type_counts.items()):
            print(f"    {ft}: {count} features")

    print(f"\nOutput tracks:")
    for name, path in sorted(result['tracks'].items()):
        size = os.path.getsize(path)
        print(f"  {name}: {path} ({size:,} bytes)")

    print("\nTo load in JBrowse 2:")
    print("  1. Sort and index BED/GFF3 with tabix:")
    for name, path in result['tracks'].items():
        if path.endswith('.bed') or path.endswith('.bed12'):
            print(f"     sort -k1,1 -k2,2n {path} | bgzip > {path}.gz")
            print(f"     tabix -p bed {path}.gz")
        elif path.endswith('.gff3'):
            print(f"     sort -k1,1 -k4,4n {path} | bgzip > {path}.gz")
            print(f"     tabix -p gff {path}.gz")
    print("  2. Convert BedGraph to BigWig with bedGraphToBigWig")
    print("  3. Add tracks to JBrowse configuration")
    print()


if __name__ == '__main__':
    main()
