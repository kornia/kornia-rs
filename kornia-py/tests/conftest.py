"""Shared test fixtures."""

import numpy as np


def nv12_frame(w: int, h: int, rng: np.random.Generator) -> np.ndarray:
    """A synthetic tightly-packed NV12 buffer (w*h luma + w*h/2 chroma)."""
    return rng.integers(0, 256, (w * h * 3 // 2,), dtype=np.uint8)
