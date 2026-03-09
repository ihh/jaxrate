# API Reference

## Types

### `Nonterminal`

```python
class Nonterminal(NamedTuple):
    name: str
    fan_out: int = 1
    max_span: Optional[int] = None
```

A grammar nonterminal. `fan_out` is 1 for HMM/SCFG nonterminals, 2 for MCFG nonterminals that generate two non-contiguous spans (e.g. pseudoknot stems). `max_span` optionally constrains the maximum span of each component.

### `EmissionGroup`

```python
class EmissionGroup(NamedTuple):
    n_positions: int
    model_index: int
```

Specifies how many alignment columns a rule emits and which substitution model to use.

- `n_positions=1`: single-column emission (unpaired)
- `n_positions=2`: two-column emission (base pair)
- `model_index`: index into the `TerminalWeights` arrays

### `Rule`

```python
class Rule(NamedTuple):
    lhs: int
    rhs: tuple
    emissions: tuple
    log_weight: float
    composition: tuple = ()
```

A production rule. `lhs` is the nonterminal index on the left-hand side. `rhs` is a tuple of nonterminal indices on the right-hand side (empty for terminal rules). `emissions` is a tuple of `EmissionGroup` values. `log_weight` is the log-probability of the rule.

### `Grammar`

```python
class Grammar(NamedTuple):
    nonterminals: list
    rules: list
    start: int = 0
    n_models: int = 1
```

A stochastic grammar with a list of `Nonterminal` values, a list of `Rule` values, a start nonterminal index, and the number of emission models used.

### `CompiledGrammar`

```python
class CompiledGrammar(NamedTuple):
    grammar_class: str       # 'hmm', 'scfg', or 'mcfg'
    n_nonterminals: int
    n_rules: int
    rule_lhs: jnp.ndarray          # (n_rules,) int32
    rule_rhs: jnp.ndarray          # (n_rules, max_rhs) int32, -1 = padding
    rule_n_rhs: jnp.ndarray        # (n_rules,) int32
    rule_log_weights: jnp.ndarray  # (n_rules,) float64
    rule_emission_model: jnp.ndarray  # (n_rules, max_emit) int32
    rule_emission_npos: jnp.ndarray   # (n_rules, max_emit) int32
    rule_n_emissions: jnp.ndarray     # (n_rules,) int32
    rules_for_nt: jnp.ndarray      # (n_nonterminals, max_rules_per_nt) int32
    n_rules_for_nt: jnp.ndarray    # (n_nonterminals,) int32
    start: int
    n_models: int
    fan_outs: jnp.ndarray           # (n_nonterminals,) int32
    max_spans: jnp.ndarray = None   # (n_nonterminals,) int32, -1 = unlimited
```

Dense array representation of a `Grammar`, produced by `compile_grammar()`. Used by all algorithm implementations.

### `TerminalWeights`

```python
class TerminalWeights(NamedTuple):
    single: jnp.ndarray              # (K, C) float64
    paired: Optional[jnp.ndarray] = None  # (K, C, C) float64
    kmer: Optional[jnp.ndarray] = None    # (K, C_kmer) float64
    C: int = 0
```

Pre-computed phylogenetic log-likelihoods for each alignment column (or column pair) under each emission model.

- `single[k, c]`: log-likelihood of column `c` under model `k`
- `paired[k, i, j]`: log-likelihood of column pair `(i, j)` under model `k`
- `C`: number of alignment columns

### `ParseTree`

```python
class ParseTree(NamedTuple):
    labels: jnp.ndarray             # (C,) int32
    log_prob: float
    rule_trace: Optional[list] = None
```

### `TrainState`

```python
class TrainState(NamedTuple):
    log_weights: jnp.ndarray  # (n_rules,) float64
    iteration: int = 0
    log_likelihood: float = float('-inf')
```

## Grammar construction

### `GrammarBuilder`

```python
class GrammarBuilder:
    def add_nonterminal(self, name, fan_out=1, max_span=None) -> int
    def add_rule(self, lhs, rhs=(), emissions=(), log_weight=0.0, composition=()) -> int
    def build(self, start=0, n_models=1) -> Grammar
```

Programmatic grammar construction. `add_nonterminal` returns the nonterminal index. `add_rule` returns the rule index. `build` validates and returns the `Grammar`.

### `compile_grammar`

```python
def compile_grammar(grammar: Grammar) -> CompiledGrammar
```

Compile a `Grammar` into dense JAX arrays. Auto-classifies as `'hmm'`, `'scfg'`, or `'mcfg'`.

### `classify_grammar`

```python
def classify_grammar(grammar: Grammar) -> str
```

Returns `'hmm'`, `'scfg'`, or `'mcfg'` based on fan-out and rule structure.

### `validate_grammar`

```python
def validate_grammar(grammar: Grammar) -> None
```

Check grammar consistency. Raises `ValueError` on invalid nonterminal references, `max_span < 1`, or other errors.

## Algorithms

All top-level algorithm functions accept either a `Grammar` or `CompiledGrammar` and a `TerminalWeights`. They auto-dispatch to the appropriate implementation based on grammar class.

### `inside`

```python
def inside(grammar, terminal_weights) -> (chart, log_likelihood)
```

Compute inside (forward) probabilities.

- **HMM**: `chart` is `(C, K)` log forward probabilities. O(CK²).
- **SCFG**: `chart` is `(K, C+1, C+1)` inside chart. O(C³K³).
- **MCFG**: `chart` is `(alpha1, alpha2)` where `alpha1` is `(K, C+1, C+1)` and `alpha2` is `(K, C+1, C+1, C+1, C+1)`. O(C⁶K³) or O(C²L²K³ + C³K³) with `max_span`.

### `outside`

```python
def outside(grammar, terminal_weights, inside_chart=None) -> (chart, log_likelihood)
```

Compute outside (backward) probabilities. Supported for HMM and SCFG. MCFG raises `NotImplementedError`.

### `viterbi`

```python
def viterbi(grammar, terminal_weights) -> (labels, log_prob)
```

Find the most probable parse. Returns per-column nonterminal labels `(C,) int32` and the log-probability of the best parse.

## Low-level algorithm functions

### HMM

```python
from jaxrate.hmm import hmm_forward, hmm_backward, hmm_viterbi, hmm_posteriors

hmm_forward(cg, tw) -> (alpha, log_likelihood)
hmm_backward(cg, tw) -> (beta, log_likelihood)
hmm_viterbi(cg, tw) -> (labels, log_prob)
hmm_posteriors(cg, tw) -> (posteriors, log_likelihood)
```

### SCFG

```python
from jaxrate.scfg import scfg_inside, scfg_outside, scfg_viterbi

scfg_inside(cg, tw) -> (alpha, log_likelihood)
scfg_outside(cg, tw, alpha) -> beta
scfg_viterbi(cg, tw) -> (labels, log_prob, backpointers)
```

### MCFG

```python
from jaxrate.mcfg import mcfg_inside, mcfg_viterbi

mcfg_inside(cg, tw) -> (alpha1, alpha2, log_likelihood)
mcfg_viterbi(cg, tw) -> (labels, log_prob, backpointers)
```

The `max_span` constraint is read from `cg.max_spans` and applied automatically. Fan-out 2 nonterminals with `max_span=L` have each component's span bounded by `L`, reducing complexity from O(C⁶K³) to O(C²L²K³ + C³K³).

## Training

### `train`

```python
def train(grammar, terminal_weights, n_iterations=100, convergence_tol=1e-6)
    -> (trained_grammar, TrainState, history)
```

Run EM training. Returns the optimized grammar, final `TrainState`, and a list of `(iteration, log_likelihood)` history.

### `em_step`

```python
def em_step(grammar, terminal_weights) -> (new_grammar, log_likelihood)
```

Single EM iteration: E-step (inside-outside for expected counts) then M-step (normalize per LHS nonterminal).

## Terminal weights

### `precompute_terminal_weights`

```python
def precompute_terminal_weights(alignment, tree, models,
                                paired_models=None, kmer_models=None,
                                maxChunkSize=128) -> TerminalWeights
```

Bridge to [subby](https://github.com/ihh/subby) for phylogenetic log-likelihood computation. Computes per-column (and optionally per-column-pair) log-likelihoods under each substitution model.

### Direct construction

```python
tw = TerminalWeights(
    single=jnp.array([...]),  # (K, C)
    paired=jnp.array([...]),  # (K, C, C) or None
    C=100,
)
```

## Preset grammars

### `pfold_grammar`

```python
from jaxrate.presets import pfold_grammar
grammar = pfold_grammar()
```

PFOLD-like RNA secondary structure SCFG. Nonterminals: S (structure), L (stem), F (loop). Models: 0 = unpaired, 1 = paired. Complexity: O(C³K³).

### `gene_finder_grammar`

```python
from jaxrate.presets import gene_finder_grammar
grammar = gene_finder_grammar()
```

Gene-finding HMM. States: IG (intergenic), E1/E2/E3 (exon codon positions), I (intron). Models: 0 = intergenic, 1 = exon, 2 = intron. Complexity: O(CK²).

### `pseudoknot_grammar`

```python
from jaxrate.presets import pseudoknot_grammar
grammar = pseudoknot_grammar(max_span=None)
```

Pseudoknot-capable MCFG with fan-out 2. Nonterminals: S (fan-out 1), PK (fan-out 2), L (fan-out 1). Models: 0 = unpaired, 1 = paired. `max_span` constrains each PK component's span, reducing complexity from O(C⁶K³) to O(C²L²K³ + C³K³).

## Simulation

### `simulate_parse`

```python
def simulate_parse(grammar, seq_length, key) -> (labels, rules_used)
```

Sample a derivation from the grammar using a JAX PRNG key. Returns per-column labels and a list of `(rule_index, span)` tuples.
