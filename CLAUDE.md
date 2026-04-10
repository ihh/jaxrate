# jaxrate

Stochastic MCFGs with phylogenetic terminal weights. Named after [xrate](https://github.com/ihh/dart).

## Overview

jaxrate implements grammar-based phylogenetic annotation using JAX. It supports:
- **HMMs** (right-linear grammars) — O(CK²) scan-based forward/backward/Viterbi
- **SCFGs** (context-free grammars) — O(C³K³) chart-based inside/outside/CYK
- **MCFGs** (multiple context-free grammars, fan-out 2) — O(C⁶K³) for pseudoknots
  - With `max_span` constraint: O(C²L²K³ + C³K³)

Takes [subby](https://github.com/ihh/subby) as a dependency for computing phylogenetic terminal weights (per-column log-likelihoods from alignment + tree + substitution model).

## Architecture

```
jaxrate/
  __init__.py             # Public API exports
  types.py                # Grammar, Rule, Nonterminal, etc. (NamedTuples)
  grammar.py              # Grammar construction, validation, compilation
  terminal_weights.py     # subby → terminal weight tables
  inside.py               # General inside algorithm (dispatches by grammar class)
  outside.py              # General outside algorithm
  viterbi.py              # CYK/Viterbi decoding
  train.py                # EM training + gradient optimization
  simulate.py             # Sample from grammar
  hmm.py                  # scan-based forward/backward/Viterbi
  scfg.py                 # chart-based inside/outside/CYK
  mcfg.py                 # fan-out 2 chart (with max_span support)
  xrate_parser.py         # Parser for xrate .eg grammar files → Grammar + subby models
  nj.py                   # Neighbor-Joining tree construction + distance matrices
  presets.py              # Pre-built grammars (pfold, gene_finder, pseudoknot)
  _log_semiring.py        # logsumexp utilities (NEG_INF = -1e38)
  data/
    RF00390.sto           # Rfam pseudoknot alignment (TYMV_upPK, 23 nt, 7 seqs)
    pfold.eg              # xrate pfold grammar (RNA SCFG with single + paired chains)
    nullrna.eg            # xrate null RNA grammar (simple HMM)

tests/
  conftest.py             # Fixtures: simple_terminal_weights, paired_terminal_weights
  test_grammar.py         # GrammarBuilder, classify, validate, compile, presets
  test_hmm.py             # Forward, backward, Viterbi, posteriors
  test_scfg.py            # Inside, outside, CYK, paired emissions
  test_mcfg.py            # MCFG inside, Viterbi, composition rules
  test_max_span.py        # max_span constraint correctness
  test_xrate_parser.py    # xrate .eg parser: S-expr, nullrna, pfold, subby conversion
  test_nj.py              # Neighbor-Joining: tree construction, distances, RF00390
  test_cross_algorithm.py # HMM forward == SCFG inside for right-linear grammars

examples/
  gene_finding.py         # HMM gene finder with synthetic weights
  rna_structure_prediction.py  # SCFG RNA structure with synthetic weights
  rna_structure_phylogenetic.py # Real phylo: pfold.eg + NJ tree + subby → Viterbi
  pseudoknot_prediction.py     # MCFG pseudoknot with RF00390 data
```

## Public API

```python
# Types (all NamedTuples for JAX pytree compatibility)
Nonterminal(name, fan_out=1, max_span=None)
EmissionGroup(n_positions, model_index)
Rule(lhs, rhs, emissions, log_weight, composition=())
Grammar(nonterminals, rules, start=0, n_models=1)
CompiledGrammar(...)      # Dense JAX arrays for JIT
TerminalWeights(single, paired=None, kmer=None, C=0)
ParseTree(labels, log_prob, rule_trace=None)
TrainState(log_weights, iteration=0, log_likelihood=-inf)

# Grammar construction
GrammarBuilder()          # Programmatic builder
compile_grammar(grammar)  # → CompiledGrammar
classify_grammar(grammar) # → 'hmm' | 'scfg' | 'mcfg'
validate_grammar(grammar) # raises ValueError

# High-level algorithms (auto-dispatch by grammar class)
inside(grammar, terminal_weights)  # → (chart, log_likelihood)
outside(grammar, terminal_weights) # → (chart, log_likelihood)
viterbi(grammar, terminal_weights) # → (labels, log_prob)

# Terminal weights (bridge to subby)
precompute_terminal_weights(alignment, tree, models,
    paired_models=None, kmer_models=None)  # → TerminalWeights

# Training
train(grammar, terminal_weights, n_iterations=100)
em_step(grammar, terminal_weights) # → (new_grammar, log_likelihood)

# xrate parser
parse_xrate(text, start_nonterminal=None)  # → XrateGrammar
parse_xrate_file(filepath)                 # → XrateGrammar
XrateGrammar.grammar              # jaxrate Grammar
XrateGrammar.chains               # list of chain dicts (pi, rate_matrix, terminals)
XrateGrammar.to_subby_models()    # → {'single': [...], 'paired': [...]}

# Neighbor-Joining tree construction
neighbor_joining(distance_matrix, leaf_names=None)  # → dict with parentIndex, distanceToParent, newick
to_subby_tree(nj_result)                            # → subby Tree NamedTuple
hamming_distances(alignment)                        # → (N, N) pairwise Hamming distances
jukes_cantor_distances(alignment, A=4)              # → (N, N) JC-corrected distances

# Presets
pfold_grammar()                   # RNA structure (SCFG)
gene_finder_grammar()             # Gene finding (HMM)
pseudoknot_grammar(max_span=None) # Pseudoknots (MCFG)
```

## Conventions

- All NamedTuples for JAX pytree compatibility
- Log-space arithmetic throughout (NEG_INF = -1e38, not -inf, to avoid NaN gradients)
- `jax.lax.scan` for HMM algorithms
- `jax.lax.fori_loop` for SCFG/MCFG chart filling
- Column indices are 0-based
- Grammar classification: 'hmm', 'scfg', 'mcfg'
- Dispatcher pattern: inside/outside/viterbi auto-dispatch to algorithm-specific implementations
- EmissionGroup.n_positions: 1=single column, 2=paired columns, 3=codon

## Testing

```bash
source ~/jax-env/bin/activate
cd ~/jaxrate
pip install -e ".[dev]"
pytest -v
```

## Dependencies

- Python >= 3.10
- JAX >= 0.4.20
- subby (phylogenetic substitution models)

---

## Roadmap: Next implementation tasks

### 1. ~~xrate grammar file parser~~ ✅ DONE

**Goal**: Parse xrate `.eg` grammar files (S-expression format) from [dart/grammars/](https://github.com/ihh/dart/tree/master/grammars) and convert them to jaxrate Grammar + subby substitution models.

**xrate format reference**: https://github.com/ihh/dart/blob/master/doc/XrateFormat.txt

**Key xrate concepts to support**:
- **Alphabet**: `(alphabet (name RNA) (token (a c g u)) (complement (u g c a)) ...)` — defines terminal symbols
- **Chains** (substitution models): `(chain (terminal (NUC)) (update-policy rev) (initial (state (a)) (prob 0.25)) (mutate (from (a)) (to (c)) (rate 0.3)) ...)` — continuous-time Markov chain with initial distribution and rate matrix. These map directly to subby's substitution model format
- **Paired chains**: `(chain (terminal (LNUC RNUC)) ...)` — 16-state model for co-evolving base pairs (A²×A² state space)
- **Production rules**: Three types:
  - `(transform (from (S)) (to (S*)) (prob 0.9))` — null/structural transitions
  - `(transform (from (S*)) (to (NUC S)) (prob 0.5))` — emit + continue (left-emit)
  - `(transform (from (F)) (to (LNUC F* RNUC)))` — paired emission (left+right emit)
  - `(transform (from (B)) (to (S S)))` — bifurcation
- **Update policies**: `rev` (reversible), `irrev` (irreversible), `parametric`

**Implementation plan**:
- New module: `jaxrate/xrate_parser.py`
- Parse S-expressions (simple recursive descent — no need for a full Lisp parser; xrate S-exprs are regular)
- Extract alphabet → token list
- Extract chains → subby substitution models (rate matrix Q + initial distribution π)
- Extract production rules → jaxrate Rules via GrammarBuilder
- Map xrate pseudoterminals (NUC, LNUC/RNUC) to EmissionGroup model indices
- Return `(Grammar, list[subby_model])` tuple

**Scope**: Skip the macro/expression language (`&define`, `&foreach`, etc.) for now. Focus on:
- Literal rate/probability values
- Single and paired chains
- Standard production rules (emit, null, bifurcation)

**Key test grammars from dart** (in order of complexity):
1. `pfold.eg` — RNA SCFG with single + paired chains (the canonical test case)
2. `pfold-stemmodel.eg` — just the 16×16 paired chain (substitution model only)
3. `nullrna.eg` — simple null model
4. Other grammars in `grammars/` directory

**Tests**: Parse each grammar, verify round-trip (parsed Grammar classifies correctly, chain parameters match), run inside/viterbi on parsed grammar with real terminal weights.

### 2. ~~Neighbor-Joining tree construction~~ ✅ DONE

**Goal**: Implement the Neighbor-Joining (NJ) algorithm in JAX so users can construct phylogenetic trees from distance matrices computed from their alignments, without needing an external tree.

**Where**: Could be a new module in subby (`subby/jax/nj.py`) or a separate `jaxtree` package. The user mentioned "jaxtree" — if creating a new repo, follow the same conventions as jaxrate/subby.

**Input**: distance matrix `(N, N)` from pairwise sequence comparison
**Output**: tree topology + branch lengths in Newick or subby's tree format

**Use case**: Many users have alignments but no tree. NJ gives a quick-and-dirty tree from Hamming/JC distances, enabling the full jaxrate pipeline.

### 3. ~~Real phylogenetic examples~~ ✅ DONE

**Goal**: Replace synthetic terminal weights in examples with real phylogenetic computations using actual trees and substitution models.

**What "real" means**:
- Load a Stockholm-format MSA (e.g., from Rfam)
- Construct or load a phylogenetic tree (Newick format, or NJ from alignment distances)
- Define substitution models: use xrate-parsed chains (pfold single + paired models) or manually specified rate matrices
- Call `precompute_terminal_weights(alignment, tree, models)` to get real phylogenetic terminal weights
- Run grammar algorithms on real data

**Concrete example**: RNA structure prediction with pfold
1. Parse `pfold.eg` to get Grammar + substitution models
2. Load an Rfam alignment (RF00390 already bundled, or fetch others)
3. Build NJ tree from alignment distances (or use a provided tree)
4. Compute terminal weights via subby
5. Run Viterbi → predicted RNA secondary structure
6. Compare to Rfam SS_cons annotation

**Another example**: Gene finding with real genomic data
- Use a real coding/non-coding substitution model
- Apply to a multi-species alignment of a known gene region

### 4. ~~Integrated rate matrix fitting~~ ✅ DONE

**Goal**: Joint EM training of grammar rule weights and substitution model rate matrices, matching xrate's full training loop. Users should not need to call subby directly — jaxrate wraps it.

**Key design decisions**:
- **`PhyloModel`** class wraps alignment + tree + substitution models (rate matrices Q and initial distributions π). Provides a `.terminal_weights()` method that calls subby internally, so callers never import or configure subby themselves.
- **`phylo_train()`** runs the integrated EM loop: E-step (inside-outside → expected rule usage counts + per-model column posteriors), M-step (re-estimate rule weights + rate matrices + π from expected counts), recompute terminal weights, repeat.
- **Fit control**: `fit_rules=True`, `fit_rates=True`, `fit_pi=True` flags let callers disable parts of the M-step (e.g., fix rate matrices and only train rule weights, or vice versa).
- **Rate matrix M-step**: Uses expected column counts (posterior probability of each model at each column) as weights for a weighted expected-substitution-count update. For reversible models, the symmetrized rate matrix is recovered via `Q_ij = n_ij / (pi_j * T)` where `n_ij` are the weighted expected substitution counts and `T` is the total expected time.

**Modules**:
- `jaxrate/phylo_model.py` — `PhyloModel` class, `phylo_train()`, `phylo_em_step()`
- Updated `jaxrate/train.py` — proper expected count computation for HMM rules
- Updated `jaxrate/__init__.py` — exports `PhyloModel`, `phylo_train`

**Tests**: `tests/test_phylo_model.py` — PhyloModel construction, terminal weight computation, integrated training convergence, fit flag control.

**Tutorial**: `docs/tutorial.md` — new section on building, fitting, and using a phylo-HMM for protein MSAs with per-state rate matrices.

### 5. Additional xrate grammar imports

After the parser works on pfold.eg, progressively import more complex grammars:
- `codon-models/` — codon substitution models (requires k-mer terminal weights)
- `hky.eg`, `jc.eg` — standard nucleotide models
- Protein models if applicable

### 6. MCFG outside algorithm

Currently raises `NotImplementedError` in `outside.py`. Needed for:
- Full EM training of MCFG grammars
- Posterior probability computation for pseudoknot prediction

---

## xrate format quick reference

xrate grammar files use nested S-expressions (Lisp-like). Key structures:

```scheme
;; Alphabet
(alphabet
 (name RNA)
 (token (a c g u))
 (complement (u g c a))
 (extend (to n) (from a) (from c) (from g) (from u)))

;; Grammar wrapper
(grammar
 (name pfold)

 ;; Substitution model (single nucleotide)
 (chain
  (update-policy rev)
  (terminal (NUC))
  (initial (state (a)) (prob 0.25))
  (mutate (from (a)) (to (c)) (rate 0.3))
  ...)

 ;; Substitution model (paired nucleotides, 16 states)
 (chain
  (update-policy rev)
  (terminal (LNUC RNUC))
  (initial (state (a a)) (prob 0.001167))
  (mutate (from (a a)) (to (c a)) (rate 0.42))
  ...)

 ;; Production rules
 (transform (from (S)) (to (S*)) (prob 0.9))       ;; null transition
 (transform (from (S*)) (to (NUC S)) (prob 0.5))   ;; left-emit single
 (transform (from (F)) (to (LNUC F* RNUC)))        ;; paired emit
 (transform (from (B)) (to (S S)))                  ;; bifurcation
 (transform (from (S)) (to ()) (prob 0.1))          ;; termination
)
```

**Mapping xrate → jaxrate**:
- xrate `(chain (terminal (NUC)) ...)` → subby model with rate matrix Q, initial π → `model_index=0` in EmissionGroup
- xrate `(chain (terminal (LNUC RNUC)) ...)` → subby paired model → `model_index=1` in EmissionGroup with `n_positions=2`
- xrate `(transform (from (A)) (to (NUC B)))` → `Rule(lhs=A, rhs=(B,), emissions=(EmissionGroup(1, 0),), log_weight=log(prob))`
- xrate `(transform (from (A)) (to (LNUC B RNUC)))` → `Rule(lhs=A, rhs=(B,), emissions=(EmissionGroup(2, 1),), log_weight=log(prob))`
- xrate `(transform (from (A)) (to (B C)))` → `Rule(lhs=A, rhs=(B, C), emissions=(), log_weight=log(prob))`
- xrate `(transform (from (A)) (to (B)))` → `Rule(lhs=A, rhs=(B,), emissions=(), log_weight=log(prob))`
