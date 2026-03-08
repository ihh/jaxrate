# jaxrate

Stochastic MCFGs with phylogenetic terminal weights. Named after xrate.

## Overview

jaxrate implements grammar-based phylogenetic annotation using JAX. It supports:
- **HMMs** (right-linear grammars) — O(CK²) scan-based forward/backward/Viterbi
- **SCFGs** (context-free grammars) — O(C³K³) chart-based inside/outside/CYK
- **MCFGs** (multiple context-free grammars, fan-out 2) — O(C⁶K³) for pseudoknots

Takes subby as a dependency for computing phylogenetic terminal weights.

## Architecture

```
jaxrate/
  types.py              # Grammar, Rule, Nonterminal, etc. (NamedTuples)
  grammar.py            # Grammar construction, validation, compilation
  terminal_weights.py   # subby → terminal weight tables
  inside.py             # General inside algorithm (dispatches by grammar class)
  outside.py            # General outside algorithm
  viterbi.py            # CYK/Viterbi decoding
  train.py              # EM training + gradient optimization
  simulate.py           # Sample from grammar
  hmm.py                # scan-based forward/backward/Viterbi
  scfg.py               # chart-based inside/outside/CYK
  mcfg.py               # fan-out 2 chart
  presets.py            # Pre-built grammars
  _log_semiring.py      # logsumexp utilities
```

## Conventions

- All NamedTuples for JAX pytree compatibility
- Log-space arithmetic throughout
- `jax.lax.scan` for HMM algorithms
- `jax.lax.fori_loop` for SCFG/MCFG chart filling
- Column indices are 0-based
- Grammar classification: 'hmm', 'scfg', 'mcfg'

## Testing

```bash
source ~/jax-env/bin/activate
cd ~/jaxrate
pip install -e ".[dev]"
pytest -v
```
