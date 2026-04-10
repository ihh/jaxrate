# Tutorial

## Step 1: Define a grammar

jaxrate grammars are built using `GrammarBuilder`:

```python
from jaxrate import GrammarBuilder, EmissionGroup
import math

gb = GrammarBuilder()

# Add nonterminals
S = gb.add_nonterminal('S')          # fan-out 1 (default)
PK = gb.add_nonterminal('PK', fan_out=2)  # fan-out 2 for pseudoknots

# Add rules
gb.add_rule(S, rhs=[S], emissions=[EmissionGroup(1, 0)], log_weight=math.log(0.8))
gb.add_rule(S, rhs=[], emissions=[EmissionGroup(1, 0)], log_weight=math.log(0.2))

grammar = gb.build(start=S, n_models=1)
```

### Rule anatomy

A `Rule` has:

- **lhs**: Left-hand side nonterminal index
- **rhs**: Tuple of right-hand side nonterminal indices
- **emissions**: Tuple of `EmissionGroup(n_positions, model_index)`
- **log_weight**: Log-probability of the rule

### Grammar classification

jaxrate auto-classifies grammars:

- **HMM**: All rules have ≤1 RHS nonterminal (right-linear)
- **SCFG**: All nonterminals have fan-out 1, but some rules are binary
- **MCFG**: Any nonterminal has fan-out > 1

## Step 2: Pre-compute terminal weights

Terminal weights are phylogenetic log-likelihoods computed once upfront using subby:

```python
from jaxrate import precompute_terminal_weights
from subby.jax import LogLike
from subby.jax.models import jukes_cantor_model

# Using subby directly
model_slow = jukes_cantor_model(4)
model_fast = scale_model(model_slow, 2.0)

tw = precompute_terminal_weights(
    alignment, tree,
    models=[model_slow, model_fast],
)
# tw.single has shape (2, C) — log-likelihoods under each model per column
```

Or construct directly:

```python
from jaxrate import TerminalWeights
import jax.numpy as jnp

tw = TerminalWeights(
    single=jnp.array([slow_ll, fast_ll]),  # (K, C)
    paired=paired_ll,                       # (K_p, C, C) or None
    C=100,
)
```

## Step 3: Run algorithms

```python
from jaxrate import inside, outside, viterbi

# Inside (forward) — total log-likelihood
chart, ll = inside(grammar, tw)

# Viterbi — best annotation
labels, log_prob = viterbi(grammar, tw)

# Outside (backward)
beta, ll = outside(grammar, tw)
```

## Step 4: Use preset grammars

```python
from jaxrate.presets import pfold_grammar, gene_finder_grammar, pseudoknot_grammar

# RNA structure prediction (SCFG)
rna_grammar = pfold_grammar()

# Gene finding (HMM)
gene_grammar = gene_finder_grammar()

# Pseudoknot prediction (MCFG)
pk_grammar = pseudoknot_grammar()

# Pseudoknot with bounded stem span (reduces complexity)
pk_grammar_bounded = pseudoknot_grammar(max_span=15)
```

## Step 5: EM training

```python
from jaxrate import train

trained_grammar, state, history = train(
    grammar, tw,
    n_iterations=100,
    convergence_tol=1e-6,
)
print(f"Converged at iteration {state.iteration}, LL={state.log_likelihood:.4f}")
```

## Step 6: Pseudoknot prediction with max_span

The `pseudoknot_grammar` produces an MCFG with fan-out 2 nonterminals. The `max_span` parameter constrains each component's span, reducing complexity from O(C⁶K³) to O(C²L²K³ + C³K³):

```python
from jaxrate.presets import pseudoknot_grammar
from jaxrate.mcfg import mcfg_inside, mcfg_viterbi
from jaxrate import compile_grammar, TerminalWeights

# Build grammar with bounded pseudoknot stems
grammar = pseudoknot_grammar(max_span=15)
cg = compile_grammar(grammar)

# Run Viterbi decoding
labels, log_prob, backpointers = mcfg_viterbi(cg, tw)

# Run inside algorithm
alpha1, alpha2, log_likelihood = mcfg_inside(cg, tw)
```

The `max_span` limits how far apart the paired positions in a pseudoknot stem can be. For example, `max_span=15` means each component of the PK nonterminal can span at most 15 columns. Setting `max_span` equal to or larger than the sequence length gives the same result as unlimited.

## Step 7: Integrated phylogenetic training with PhyloModel

`PhyloModel` wraps an alignment, tree, grammar, and substitution models into a single object. It handles subby interaction internally so you never need to import or configure it yourself.

### Building a PhyloModel

```python
import math
import numpy as np
from jaxrate import GrammarBuilder, EmissionGroup, PhyloModel

# Build a 2-state phylo-HMM
gb = GrammarBuilder()
SLOW = gb.add_nonterminal('slow')
FAST = gb.add_nonterminal('fast')
gb.add_rule(SLOW, rhs=[SLOW], emissions=[EmissionGroup(1, 0)], log_weight=math.log(0.9))
gb.add_rule(SLOW, rhs=[FAST], emissions=[EmissionGroup(1, 0)], log_weight=math.log(0.09))
gb.add_rule(SLOW, rhs=[],     emissions=[EmissionGroup(1, 0)], log_weight=math.log(0.01))
gb.add_rule(FAST, rhs=[FAST], emissions=[EmissionGroup(1, 1)], log_weight=math.log(0.9))
gb.add_rule(FAST, rhs=[SLOW], emissions=[EmissionGroup(1, 1)], log_weight=math.log(0.09))
gb.add_rule(FAST, rhs=[],     emissions=[EmissionGroup(1, 1)], log_weight=math.log(0.01))
grammar = gb.build(start=SLOW, n_models=2)

# Define per-state rate matrices (DNA, A=4)
chains = [
    {'rate_matrix': np.eye(4) * -0.5 + np.full((4,4), 0.5/3) - np.diag(np.full(4, 0.5/3)),
     'pi': np.full(4, 0.25)},  # slow
    {'rate_matrix': np.eye(4) * -3.0 + np.full((4,4), 3.0/3) - np.diag(np.full(4, 3.0/3)),
     'pi': np.full(4, 0.25)},  # fast
]

pm = PhyloModel(grammar, alignment, tree, chains)
```

### Or from an xrate grammar file

```python
from jaxrate import parse_xrate_file, PhyloModel

xg = parse_xrate_file('pfold.eg')
pm = PhyloModel.from_xrate(xg, alignment, tree)
```

### Computing terminal weights and running algorithms

```python
from jaxrate import viterbi, inside

# PhyloModel computes terminal weights internally (calls subby under the hood)
tw = pm.terminal_weights()

# Use them with any jaxrate algorithm
labels, log_prob = viterbi(pm.grammar, tw)
chart, ll = inside(pm.grammar, tw)
```

### Integrated EM training

`phylo_train` jointly fits grammar rule weights and substitution model parameters:

```python
from jaxrate import phylo_train

# Fit everything (rules + rate matrices + equilibrium distributions)
history = phylo_train(pm, n_iterations=50, pseudocounts=1e-4)
print(f"Final LL: {history[-1][1]:.2f}")

# Check fitted rate matrices
for i, chain in enumerate(pm.chains):
    rate = -chain['rate_matrix'].diagonal().mean()
    print(f"  Model {i}: mean rate = {rate:.4f}")
```

### Controlling what gets optimized

Use `fit_rules`, `fit_rates`, and `fit_pi` flags:

```python
# Fix rate matrices, only train grammar weights
history = phylo_train(pm, n_iterations=50,
                      fit_rules=True, fit_rates=False, fit_pi=False)

# Fix grammar, only train rate matrices + pi
history = phylo_train(pm, n_iterations=50,
                      fit_rules=False, fit_rates=True, fit_pi=True,
                      pseudocounts=1e-4)
```

### Pseudocounts

The `pseudocounts` parameter adds regularization to prevent rate matrix degeneration on sparse data (following xrate convention). It adds pseudo-wait-time to the expected dwell times in the M-step. xrate's default is `1e-4`. For well-conditioned data (many sequences, long branches), `pseudocounts=0` works fine.

### Rate matrix M-step details

The rate matrix update follows xrate's approach:
- **Off-diagonal rates**: Q_ij = (expected i→j transitions) / (expected dwell time in state i)
- **Equilibrium distribution**: for reversible models, π is the stationary distribution of the new Q (solving π·Q = 0). This ensures detailed balance.
- **Expected counts** come from subby's phylogenetic inside-outside on the tree, weighted by grammar state posteriors from the HMM/SCFG forward-backward.
