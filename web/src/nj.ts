/**
 * Neighbor-Joining tree construction and distance matrices.
 * Mirrors jaxrate/nj.py.
 */

// ── Types ───────────────────────────────────────────────────────────

export interface NJResult {
  parentIndex: Int32Array;      // (2N-1,) — parent of each node (-1 for root)
  distanceToParent: Float64Array; // (2N-1,) — branch length to parent
  leafNames: string[];
  newick: string;
  nLeaves: number;
  nNodes: number;
}

// ── Distance matrices ───────────────────────────────────────────────

/**
 * Pairwise Hamming distances from alignment.
 * @param alignment (N, L) — rows are sequences, columns are sites
 * @param gapToken token index for gap character (excluded from comparison)
 */
export function hammingDistances(
  alignment: number[][], gapToken = -1, normalize = true
): Float64Array {
  const N = alignment.length;
  const L = alignment[0].length;
  const dist = new Float64Array(N * N);

  for (let i = 0; i < N; i++) {
    for (let j = i + 1; j < N; j++) {
      let diffs = 0;
      let valid = 0;
      for (let c = 0; c < L; c++) {
        if (alignment[i][c] === gapToken || alignment[j][c] === gapToken) continue;
        valid++;
        if (alignment[i][c] !== alignment[j][c]) diffs++;
      }
      const d = valid > 0 ? (normalize ? diffs / valid : diffs) : 0;
      dist[i * N + j] = d;
      dist[j * N + i] = d;
    }
  }
  return dist;
}

/**
 * Jukes-Cantor corrected distances.
 * d_JC = -(A-1)/A × ln(1 - A/(A-1) × p)
 */
export function jukesCantor(
  alignment: number[][], A = 4, gapToken = -1
): Float64Array {
  const N = alignment.length;
  const hamming = hammingDistances(alignment, gapToken, true);
  const dist = new Float64Array(N * N);
  const factor = (A - 1) / A;
  const scale = A / (A - 1);

  for (let i = 0; i < N; i++) {
    for (let j = i + 1; j < N; j++) {
      const p = hamming[i * N + j];
      const arg = 1 - scale * p;
      const d = arg > 0 ? -factor * Math.log(arg) : 10.0;  // cap at 10
      dist[i * N + j] = d;
      dist[j * N + i] = d;
    }
  }
  return dist;
}

// ── Neighbor-Joining algorithm ──────────────────────────────────────

export function neighborJoining(
  distMatrix: Float64Array, N: number,
  leafNames?: string[]
): NJResult {
  const names = leafNames ?? Array.from({ length: N }, (_, i) => `leaf_${i}`);
  const totalNodes = 2 * N - 1;

  const parentIndex = new Int32Array(totalNodes).fill(-1);
  const distanceToParent = new Float64Array(totalNodes);

  // Working distance matrix (expandable)
  const d = new Map<string, number>();
  const active = new Set<number>();
  for (let i = 0; i < N; i++) active.add(i);

  const key = (a: number, b: number) => `${Math.min(a, b)},${Math.max(a, b)}`;
  for (let i = 0; i < N; i++) {
    for (let j = i + 1; j < N; j++) {
      d.set(key(i, j), distMatrix[i * N + j]);
    }
  }

  let nextNode = N;
  const nodeNames: string[] = [...names];

  while (active.size > 2) {
    const nodes = [...active];
    const n = nodes.length;

    // Compute r_i = sum of distances to all other active nodes
    const r = new Map<number, number>();
    for (const i of nodes) {
      let sum = 0;
      for (const j of nodes) {
        if (i !== j) sum += d.get(key(i, j)) ?? 0;
      }
      r.set(i, sum);
    }

    // Find pair (i,j) minimizing Q(i,j) = (n-2)*d(i,j) - r_i - r_j
    let bestQ = Infinity;
    let bestI = nodes[0];
    let bestJ = nodes[1];

    for (let a = 0; a < n; a++) {
      for (let b = a + 1; b < n; b++) {
        const i = nodes[a];
        const j = nodes[b];
        const q = (n - 2) * (d.get(key(i, j)) ?? 0) - (r.get(i) ?? 0) - (r.get(j) ?? 0);
        if (q < bestQ) {
          bestQ = q;
          bestI = i;
          bestJ = j;
        }
      }
    }

    // Create new internal node
    const u = nextNode++;
    nodeNames.push(`internal_${u}`);
    const dij = d.get(key(bestI, bestJ)) ?? 0;
    const ri = r.get(bestI) ?? 0;
    const rj = r.get(bestJ) ?? 0;

    const diu = dij / 2 + (n > 2 ? (ri - rj) / (2 * (n - 2)) : 0);
    const dju = dij - diu;

    parentIndex[bestI] = u;
    parentIndex[bestJ] = u;
    distanceToParent[bestI] = Math.max(0, diu);
    distanceToParent[bestJ] = Math.max(0, dju);

    // Compute distances from u to all other active nodes
    for (const k of nodes) {
      if (k === bestI || k === bestJ) continue;
      const dik = d.get(key(bestI, k)) ?? 0;
      const djk = d.get(key(bestJ, k)) ?? 0;
      const duk = (dik + djk - dij) / 2;
      d.set(key(u, k), Math.max(0, duk));
    }

    active.delete(bestI);
    active.delete(bestJ);
    active.add(u);
  }

  // Connect final two nodes
  if (active.size === 2) {
    const [a, b] = [...active];
    const dab = d.get(key(a, b)) ?? 0;
    // Make the higher-indexed node the root's parent
    const root = nextNode > totalNodes - 1 ? b : Math.max(a, b);
    const child = a === root ? b : a;
    parentIndex[child] = root;
    distanceToParent[child] = dab;
  }

  // Build Newick string
  const newick = buildNewick(parentIndex, distanceToParent, nodeNames, N, totalNodes);

  return {
    parentIndex, distanceToParent,
    leafNames: names, newick,
    nLeaves: N, nNodes: totalNodes,
  };
}

function buildNewick(
  parentIndex: Int32Array, distanceToParent: Float64Array,
  nodeNames: string[], nLeaves: number, nNodes: number
): string {
  // Find root (node with parent -1)
  let root = -1;
  for (let i = 0; i < nNodes; i++) {
    if (parentIndex[i] === -1) { root = i; break; }
  }
  if (root === -1) root = nNodes - 1;

  // Build children map
  const children = new Map<number, number[]>();
  for (let i = 0; i < nNodes; i++) {
    if (parentIndex[i] >= 0) {
      const p = parentIndex[i];
      if (!children.has(p)) children.set(p, []);
      children.get(p)!.push(i);
    }
  }

  function toNewick(node: number): string {
    const kids = children.get(node) ?? [];
    if (kids.length === 0) {
      return `${nodeNames[node]}:${distanceToParent[node].toFixed(6)}`;
    }
    const kidStrs = kids.map(toNewick).join(',');
    const dist = parentIndex[node] >= 0 ? `:${distanceToParent[node].toFixed(6)}` : '';
    return `(${kidStrs})${dist}`;
  }

  return toNewick(root) + ';';
}
