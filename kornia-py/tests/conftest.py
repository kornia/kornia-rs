"""Shared pytest fixtures for kornia-py tests."""

import numpy as np
import pytest


@pytest.fixture(scope="session")
def rand_u8_1080p() -> np.ndarray:
    """Deterministic 1080p uint8 RGB. Seed-pinned so the perf gate
    doesn't flap on a pathological random draw."""
    rng = np.random.default_rng(0)
    return rng.integers(0, 256, (1080, 1920, 3), dtype=np.uint8)
