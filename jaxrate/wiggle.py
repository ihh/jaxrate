"""Wiggle and BedGraph track output for genome annotation results.

Produces UCSC Genome Browser-compatible track files from jaxrate
posterior probability annotations.
"""

import numpy as np


def write_wiggle(filepath, values, chrom='chr1', start=0, step=1,
                 track_name='jaxrate', description='', span=1):
    """Write a fixedStep wiggle (.wig) file.

    Args:
        filepath: output file path
        values: (C,) array of per-position values (e.g., posterior probabilities)
        chrom: chromosome/contig name
        start: 1-based start position (WIG format is 1-based)
        step: step size between values
        track_name: track name for browser display
        description: track description
        span: span of each data point
    """
    values = np.asarray(values, dtype=np.float64)
    with open(filepath, 'w') as f:
        f.write(f'track type=wiggle_0 name="{track_name}" '
                f'description="{description}"\n')
        f.write(f'fixedStep chrom={chrom} start={start} step={step}')
        if span != 1:
            f.write(f' span={span}')
        f.write('\n')
        for v in values:
            f.write(f'{v:.6f}\n')


def write_bedgraph(filepath, values, chrom='chr1', start=0,
                   track_name='jaxrate', description=''):
    """Write a BedGraph (.bedGraph) file.

    Args:
        filepath: output file path
        values: (C,) array of per-position values
        chrom: chromosome/contig name
        start: 0-based start position
        track_name: track name
        description: track description
    """
    values = np.asarray(values, dtype=np.float64)
    with open(filepath, 'w') as f:
        f.write(f'track type=bedGraph name="{track_name}" '
                f'description="{description}"\n')
        for i, v in enumerate(values):
            pos = start + i
            f.write(f'{chrom}\t{pos}\t{pos + 1}\t{v:.6f}\n')


def write_multi_wiggle(filepath, tracks, chrom='chr1', start=0, step=1):
    """Write multiple tracks to a single wiggle file.

    Args:
        filepath: output file path
        tracks: list of dicts with 'name', 'description', 'values' keys
        chrom: chromosome/contig name
        start: 1-based start position
        step: step size
    """
    with open(filepath, 'w') as f:
        for track in tracks:
            name = track['name']
            desc = track.get('description', '')
            values = np.asarray(track['values'], dtype=np.float64)
            f.write(f'track type=wiggle_0 name="{name}" '
                    f'description="{desc}"\n')
            f.write(f'fixedStep chrom={chrom} start={start} step={step}\n')
            for v in values:
                f.write(f'{v:.6f}\n')


def posteriors_to_structure_track(posteriors, grammar, structural_nts=None):
    """Convert per-column posterior probabilities to a structure probability track.

    Sums posterior probabilities over nonterminals that represent structural
    (paired/stem) regions.

    Args:
        posteriors: (C, K) posterior probabilities
        grammar: Grammar with nonterminals
        structural_nts: optional set of nonterminal names to count as
            structural. If None, uses nonterminals containing 'F' or
            'pfoldCodingF' (paired emission nonterminals).

    Returns:
        (C,) array of structure probabilities per position
    """
    posteriors = np.asarray(posteriors)
    nt_names = [nt.name for nt in grammar.nonterminals]

    if structural_nts is None:
        # Default: nonterminals with paired emissions (stem/fold)
        structural_nts = set()
        for rule in grammar.rules:
            if any(e.n_positions == 2 for e in rule.emissions):
                structural_nts.add(nt_names[rule.lhs])

    struct_indices = [i for i, name in enumerate(nt_names)
                      if name in structural_nts]

    if len(struct_indices) == 0:
        return np.zeros(posteriors.shape[0])

    return posteriors[:, struct_indices].sum(axis=1)
