# jaxrate

Stochastic MCFGs with phylogenetic terminal weights, implemented in JAX. Named after [xrate](https://github.com/ihh/dart).

## Overview

jaxrate implements grammar-based phylogenetic sequence annotation. Given a multiple sequence alignment, a phylogenetic tree, and a stochastic grammar, it computes:

- **Inside/Forward**: Total log-likelihood of the alignment under the grammar
- **Outside/Backward**: Per-position contributions to the total likelihood
- **Viterbi/CYK**: Most probable annotation (parse) of the alignment
- **EM training**: Optimize grammar rule weights from data

Grammar class is auto-detected and the appropriate algorithm is dispatched:

| Class | Fan-out | Complexity | Algorithm | Example |
|-------|---------|-----------|-----------|---------|
| HMM | 1 (right-linear) | O(CK²) | scan-based | Gene finding |
| SCFG | 1 (context-free) | O(C³K³) | chart-based | RNA structure |
| MCFG | 2 | O(C⁶K³) | chart-based | Pseudoknots |
| MCFG + max_span | 2 | O(C²L²K³ + C³K³) | chart-based | Pseudoknots (bounded) |

## Installation

```bash
pip install jaxrate
```

Or from source:

```bash
git clone git@github.com:ihh/jaxrate.git
cd jaxrate
pip install -e ".[dev]"
```

## Quick start

```python
import jax.numpy as jnp
from jaxrate import (
    GrammarBuilder, EmissionGroup, compile_grammar,
    TerminalWeights, inside, viterbi,
)

# Build a 2-state HMM
gb = GrammarBuilder()
S0 = gb.add_nonterminal('slow')
S1 = gb.add_nonterminal('fast')

# State transitions with emissions
import math
gb.add_rule(S0, rhs=[S0], emissions=[EmissionGroup(1, 0)], log_weight=math.log(0.9))
gb.add_rule(S0, rhs=[S1], emissions=[EmissionGroup(1, 0)], log_weight=math.log(0.09))
gb.add_rule(S0, rhs=[], emissions=[EmissionGroup(1, 0)], log_weight=math.log(0.01))
gb.add_rule(S1, rhs=[S1], emissions=[EmissionGroup(1, 1)], log_weight=math.log(0.9))
gb.add_rule(S1, rhs=[S0], emissions=[EmissionGroup(1, 1)], log_weight=math.log(0.09))
gb.add_rule(S1, rhs=[], emissions=[EmissionGroup(1, 1)], log_weight=math.log(0.01))

grammar = gb.build(start=S0, n_models=2)

# Terminal weights: pre-computed phylogenetic log-likelihoods
C = 10
tw = TerminalWeights(
    single=jnp.array([[-1.0]*5 + [-3.0]*5, [-3.0]*5 + [-1.0]*5]),
    C=C,
)

# Compute log-likelihood
chart, ll = inside(grammar, tw)
print(f"Log-likelihood: {ll:.4f}")

# Viterbi decoding
labels, log_prob = viterbi(grammar, tw)
print(f"Best path: {labels}")
```

## Repository layout

```
jaxrate/            Grammar-based phylogenetic annotation library
  data/             Bundled data files (Rfam alignments)
examples/           Example scripts
tests/              Unit and integration tests
docs/               Documentation source
scripts/            Build and utility scripts
```

## Dependencies

- [JAX](https://github.com/google/jax) >= 0.4.20
- [subby](https://github.com/ihh/subby) — phylogenetic sufficient statistics

## Documentation

Built documentation is at [ihh.github.io/jaxrate](https://ihh.github.io/jaxrate/). Source is in `docs/`; build with `python scripts/build_docs.py`.
