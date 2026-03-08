"""Shared test fixtures for jaxrate."""

import pytest
import jax.numpy as jnp
import numpy as np

from jaxrate.types import TerminalWeights


@pytest.fixture
def simple_terminal_weights():
    """Simple 2-model, 5-column terminal weights for testing."""
    np.random.seed(42)
    C = 5
    single = jnp.array(np.random.randn(2, C) - 1.0)
    return TerminalWeights(single=single, C=C)


@pytest.fixture
def paired_terminal_weights():
    """Terminal weights with paired models for SCFG testing."""
    np.random.seed(42)
    C = 6
    single = jnp.array(np.random.randn(2, C) - 1.0)
    paired = jnp.array(np.random.randn(2, C, C) - 2.0)
    return TerminalWeights(single=single, paired=paired, C=C)
