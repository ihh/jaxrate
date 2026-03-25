"""JBrowse track output: convert jaxrate annotations to BED, GFF3, and BedGraph.

Converts Viterbi labels, posteriors, and terminal weights into
JBrowse-browsable annotation tracks for genome-aligned MSAs.
"""

import numpy as np
from collections import defaultdict


# ── Feature extraction from Viterbi labels ──────────────────────────────────

def labels_to_features(labels, grammar, chrom='chr1', start=0, strand='.'):
    """Convert per-column Viterbi labels into feature intervals.

    Groups consecutive runs of the same nonterminal into features
    with start/end coordinates.

    Args:
        labels: (C,) int32 array of nonterminal indices from viterbi()
        grammar: Grammar (for nonterminal names)
        chrom: chromosome/contig name
        start: 0-based genomic start of the alignment
        strand: '+', '-', or '.'

    Returns:
        list of dicts with keys: chrom, start, end, name, score, strand,
        nt_index, feature_type
    """
    labels = np.asarray(labels, dtype=np.int32)
    nt_names = [nt.name for nt in grammar.nonterminals]
    C = len(labels)
    if C == 0:
        return []

    # Classify nonterminals by emission type
    nt_emit_type = _classify_nonterminals(grammar)

    features = []
    run_start = 0
    run_label = int(labels[0])

    for c in range(1, C + 1):
        cur_label = int(labels[c]) if c < C else -1
        if cur_label != run_label:
            name = nt_names[run_label]
            feat_type = nt_emit_type.get(run_label, 'other')
            features.append({
                'chrom': chrom,
                'start': start + run_start,
                'end': start + c,
                'name': name,
                'score': 0,
                'strand': strand,
                'nt_index': run_label,
                'feature_type': feat_type,
            })
            run_start = c
            run_label = cur_label

    return features


def labels_to_annotation_map(labels, grammar):
    """Map Viterbi labels to semantic annotation categories.

    Returns per-column annotation strings based on grammar type:
    - RNA structure grammars: 'paired', 'unpaired', 'null'
    - Gene finding grammars: 'exon', 'intron', 'intergenic', 'null'
    - General: nonterminal name

    Args:
        labels: (C,) int32 nonterminal labels
        grammar: Grammar

    Returns:
        list of annotation strings, one per column
    """
    labels = np.asarray(labels, dtype=np.int32)
    nt_names = [nt.name for nt in grammar.nonterminals]
    nt_emit_type = _classify_nonterminals(grammar)
    return [nt_emit_type.get(int(l), nt_names[int(l)]) for l in labels]


def _classify_nonterminals(grammar):
    """Classify nonterminals by their emission type.

    Returns:
        dict mapping nt_index -> feature_type string
    """
    nt_names = [nt.name for nt in grammar.nonterminals]
    nt_emit = {}

    for rule in grammar.rules:
        lhs = rule.lhs
        if lhs in nt_emit:
            continue
        if any(e.n_positions == 2 for e in rule.emissions):
            nt_emit[lhs] = 'paired'
        elif any(e.n_positions >= 1 for e in rule.emissions):
            # Try to infer from name
            name = nt_names[lhs].lower()
            if any(k in name for k in ('exon', 'e1', 'e2', 'e3', 'coding')):
                nt_emit[lhs] = 'exon'
            elif 'intron' in name or name == 'i':
                nt_emit[lhs] = 'intron'
            elif any(k in name for k in ('intergenic', 'ig', 'null')):
                nt_emit[lhs] = 'intergenic'
            else:
                nt_emit[lhs] = 'unpaired'
        else:
            nt_emit[lhs] = 'null'

    return nt_emit


# ── Merge features for structured output ─────────────────────────────────────

def merge_features(features, grammar=None, categories=None):
    """Merge adjacent features with related types into higher-level annotations.

    For gene-finding grammars, groups exon/intron runs into gene features.
    For RNA structure grammars, groups paired/unpaired into structure elements.

    Args:
        features: list of feature dicts from labels_to_features()
        grammar: optional Grammar (for context)
        categories: optional dict mapping feature_type -> category name.
            Default groups: {'paired': 'structure', 'unpaired': 'structure',
            'exon': 'gene', 'intron': 'gene'}

    Returns:
        list of merged feature dicts with 'children' key for sub-features
    """
    if categories is None:
        categories = {
            'paired': 'structure',
            'unpaired': 'structure',
            'exon': 'gene',
            'intron': 'gene',
        }

    merged = []
    current_group = None
    current_children = []

    for feat in features:
        cat = categories.get(feat['feature_type'])
        if cat is None:
            # Flush current group
            if current_group is not None:
                merged.append(_finalize_group(current_group, current_children))
                current_group = None
                current_children = []
            merged.append(feat)
        elif current_group is not None and current_group['category'] == cat:
            current_children.append(feat)
        else:
            if current_group is not None:
                merged.append(_finalize_group(current_group, current_children))
            current_group = {
                'category': cat,
                'chrom': feat['chrom'],
                'strand': feat['strand'],
            }
            current_children = [feat]

    if current_group is not None:
        merged.append(_finalize_group(current_group, current_children))

    return merged


def _finalize_group(group, children):
    """Create a parent feature from a group of children."""
    return {
        'chrom': group['chrom'],
        'start': children[0]['start'],
        'end': children[-1]['end'],
        'name': group['category'],
        'score': 0,
        'strand': group['strand'],
        'feature_type': group['category'],
        'children': children,
    }


# ── BED output ───────────────────────────────────────────────────────────────

def write_bed(filepath, features, format='bed6'):
    """Write features to BED format.

    Args:
        filepath: output file path
        features: list of feature dicts from labels_to_features() or
            merge_features()
        format: 'bed3', 'bed6', or 'bed12'
    """
    with open(filepath, 'w') as f:
        for feat in features:
            if format == 'bed3':
                f.write(f"{feat['chrom']}\t{feat['start']}\t{feat['end']}\n")
            elif format == 'bed6':
                f.write(f"{feat['chrom']}\t{feat['start']}\t{feat['end']}\t"
                        f"{feat['name']}\t{feat['score']}\t{feat['strand']}\n")
            elif format == 'bed12':
                _write_bed12_line(f, feat)


def _write_bed12_line(f, feat):
    """Write a single BED12 line, using children as blocks if present."""
    children = feat.get('children', [])
    chrom = feat['chrom']
    start = feat['start']
    end = feat['end']
    name = feat['name']
    score = feat.get('score', 0)
    strand = feat.get('strand', '.')

    if children:
        # Use children as blocks (e.g., exons within a gene)
        block_count = len(children)
        block_sizes = ','.join(str(c['end'] - c['start']) for c in children)
        block_starts = ','.join(str(c['start'] - start) for c in children)

        # thickStart/thickEnd: coding region (exons only)
        coding = [c for c in children if c.get('feature_type') == 'exon']
        if coding:
            thick_start = coding[0]['start']
            thick_end = coding[-1]['end']
        else:
            thick_start = start
            thick_end = start

        color = _feature_color(feat.get('feature_type', ''))
    else:
        block_count = 1
        block_sizes = str(end - start)
        block_starts = '0'
        thick_start = start
        thick_end = end
        color = _feature_color(feat.get('feature_type', ''))

    f.write(f"{chrom}\t{start}\t{end}\t{name}\t{score}\t{strand}\t"
            f"{thick_start}\t{thick_end}\t{color}\t"
            f"{block_count}\t{block_sizes}\t{block_starts}\n")


def _feature_color(feature_type):
    """Return RGB color string for a feature type."""
    colors = {
        'paired': '255,0,0',        # red for base pairs
        'unpaired': '0,0,255',      # blue for unpaired
        'exon': '0,128,0',          # green for exons
        'intron': '200,200,200',    # gray for introns
        'intergenic': '240,240,240',# light gray
        'structure': '255,100,0',   # orange for structure groups
        'gene': '0,100,0',          # dark green for genes
        'null': '200,200,200',      # gray
    }
    return colors.get(feature_type, '0,0,0')


# ── GFF3 output ──────────────────────────────────────────────────────────────

def write_gff3(filepath, features, source='jaxrate', sequence_region=None):
    """Write features to GFF3 format.

    Args:
        filepath: output file path
        features: list of feature dicts from labels_to_features() or
            merge_features()
        source: value for GFF3 source column
        sequence_region: optional (chrom, start, end) for ##sequence-region pragma
    """
    with open(filepath, 'w') as f:
        f.write('##gff-version 3\n')
        if sequence_region:
            chrom, s, e = sequence_region
            f.write(f'##sequence-region {chrom} {s + 1} {e}\n')

        feat_id = 0
        for feat in features:
            children = feat.get('children', [])
            feat_id += 1
            parent_id = f'feature_{feat_id}'

            # GFF3 is 1-based, inclusive
            gff_start = feat['start'] + 1
            gff_end = feat['end']
            gff_type = _feature_to_so_type(feat.get('feature_type', 'region'))
            score = feat.get('score', '.')
            strand = feat.get('strand', '.')
            phase = '.'

            attrs = f'ID={parent_id};Name={feat["name"]}'
            if feat.get('feature_type'):
                attrs += f';feature_type={feat["feature_type"]}'

            f.write(f"{feat['chrom']}\t{source}\t{gff_type}\t"
                    f"{gff_start}\t{gff_end}\t{score}\t{strand}\t"
                    f"{phase}\t{attrs}\n")

            for child in children:
                feat_id += 1
                child_id = f'feature_{feat_id}'
                child_type = _feature_to_so_type(
                    child.get('feature_type', 'region'))
                c_start = child['start'] + 1
                c_end = child['end']
                c_score = child.get('score', '.')
                c_strand = child.get('strand', '.')
                c_phase = _codon_phase(child)

                c_attrs = (f'ID={child_id};Parent={parent_id};'
                           f'Name={child["name"]}')
                f.write(f"{child['chrom']}\t{source}\t{child_type}\t"
                        f"{c_start}\t{c_end}\t{c_score}\t{c_strand}\t"
                        f"{c_phase}\t{c_attrs}\n")


def _feature_to_so_type(feature_type):
    """Map internal feature type to Sequence Ontology term."""
    so_map = {
        'paired': 'stem_loop',
        'unpaired': 'loop',
        'structure': 'secondary_structure',
        'exon': 'exon',
        'intron': 'intron',
        'intergenic': 'intergenic_region',
        'gene': 'gene',
        'null': 'region',
    }
    return so_map.get(feature_type, 'region')


def _codon_phase(feat):
    """Return GFF3 phase for coding features."""
    name = feat.get('name', '').lower()
    if 'e1' in name:
        return '0'
    elif 'e2' in name:
        return '1'
    elif 'e3' in name:
        return '2'
    return '.'


# ── Quantitative tracks ─────────────────────────────────────────────────────

def posteriors_to_bedgraph(filepath, posteriors, grammar, chrom='chr1',
                           start=0, track_name='jaxrate_posteriors',
                           nonterminal_groups=None):
    """Write posterior probabilities as BedGraph tracks.

    Creates one BedGraph file with the per-position posterior probability
    summed over specified nonterminal groups.

    Args:
        filepath: output file path
        posteriors: (C, K) posterior probabilities
        grammar: Grammar (for nonterminal names)
        chrom: chromosome name
        start: 0-based genomic start
        track_name: track name for display
        nonterminal_groups: optional dict mapping group_name -> list of
            nonterminal names. Default: auto-detect structural vs non-structural.
    """
    posteriors = np.asarray(posteriors)
    C, K = posteriors.shape
    nt_names = [nt.name for nt in grammar.nonterminals]

    if nonterminal_groups is None:
        nt_emit = _classify_nonterminals(grammar)
        nonterminal_groups = defaultdict(list)
        for idx, name in enumerate(nt_names):
            emit_type = nt_emit.get(idx, 'null')
            if emit_type != 'null':
                nonterminal_groups[emit_type].append(name)

    with open(filepath, 'w') as f:
        for group_name, nt_list in nonterminal_groups.items():
            indices = [i for i, n in enumerate(nt_names) if n in nt_list]
            if not indices:
                continue
            values = posteriors[:, indices].sum(axis=1)
            f.write(f'track type=bedGraph name="{track_name}_{group_name}" '
                    f'description="P({group_name})"\n')
            for c in range(C):
                pos = start + c
                f.write(f'{chrom}\t{pos}\t{pos + 1}\t{values[c]:.6f}\n')


def conservation_bedgraph(filepath, terminal_weights, chrom='chr1',
                          start=0, track_name='conservation',
                          model_index=0):
    """Write per-column phylogenetic log-likelihoods as a conservation track.

    Higher log-likelihood = more conserved (fits model better).

    Args:
        filepath: output file path
        terminal_weights: TerminalWeights
        chrom: chromosome name
        start: 0-based genomic start
        track_name: track name
        model_index: which single-column model to use
    """
    single = np.asarray(terminal_weights.single)
    values = single[model_index]
    C = len(values)

    with open(filepath, 'w') as f:
        f.write(f'track type=bedGraph name="{track_name}" '
                f'description="Phylogenetic conservation (log-likelihood)"\n')
        for c in range(C):
            pos = start + c
            f.write(f'{chrom}\t{pos}\t{pos + 1}\t{values[c]:.6f}\n')


# ── High-level multi-track output ────────────────────────────────────────────

def write_jbrowse_tracks(output_dir, labels, grammar, terminal_weights,
                         posteriors=None, chrom='chr1', start=0, strand='.',
                         prefix='jaxrate'):
    """Write a complete set of JBrowse tracks from jaxrate analysis results.

    Produces:
    - {prefix}_features.bed: BED6 annotation features from Viterbi labels
    - {prefix}_features.bed12: BED12 with merged gene/structure features
    - {prefix}_features.gff3: GFF3 hierarchical annotations
    - {prefix}_conservation.bedGraph: phylogenetic conservation scores
    - {prefix}_posteriors.bedGraph: posterior probabilities (if posteriors given)

    Args:
        output_dir: output directory path
        labels: (C,) int32 Viterbi labels
        grammar: Grammar
        terminal_weights: TerminalWeights
        posteriors: optional (C, K) posterior probabilities
        chrom: chromosome name
        start: 0-based genomic start
        strand: '+', '-', or '.'
        prefix: filename prefix

    Returns:
        dict mapping track name -> file path
    """
    import os
    os.makedirs(output_dir, exist_ok=True)

    tracks = {}

    # Feature annotations from Viterbi
    features = labels_to_features(labels, grammar, chrom=chrom,
                                  start=start, strand=strand)

    bed6_path = os.path.join(output_dir, f'{prefix}_features.bed')
    write_bed(bed6_path, features, format='bed6')
    tracks['features_bed6'] = bed6_path

    # Merged features for BED12 and GFF3
    merged = merge_features(features, grammar)
    bed12_path = os.path.join(output_dir, f'{prefix}_features.bed12')
    write_bed(bed12_path, merged, format='bed12')
    tracks['features_bed12'] = bed12_path

    gff3_path = os.path.join(output_dir, f'{prefix}_features.gff3')
    C = len(np.asarray(labels))
    write_gff3(gff3_path, merged, sequence_region=(chrom, start, start + C))
    tracks['features_gff3'] = gff3_path

    # Conservation track
    if terminal_weights.single is not None:
        cons_path = os.path.join(output_dir, f'{prefix}_conservation.bedGraph')
        conservation_bedgraph(cons_path, terminal_weights,
                              chrom=chrom, start=start)
        tracks['conservation'] = cons_path

    # Posterior probability tracks
    if posteriors is not None:
        post_path = os.path.join(output_dir, f'{prefix}_posteriors.bedGraph')
        posteriors_to_bedgraph(post_path, posteriors, grammar,
                               chrom=chrom, start=start, track_name=prefix)
        tracks['posteriors'] = post_path

    return tracks
