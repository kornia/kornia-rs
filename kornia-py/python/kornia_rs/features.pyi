"""Permissive type stubs for the ``kornia_rs.features`` submodule (``Any`` for now)."""

from typing import Any

import numpy as np

def match_smnn(
    descriptors1: np.ndarray,
    descriptors2: np.ndarray,
    th: float = 0.95,
) -> tuple[np.ndarray, np.ndarray]:
    """Mutual nearest neighbors passing the ratio test in both directions.

    The contract of ``kornia.feature.match_smnn``. ``(N, 32)`` uint8 descriptors
    use Hamming distances; ``(N, D)`` float32 descriptors use L2.

    Returns ``(dists, idxs)``: ``(K, 1)`` float32 larger ratio of each match, and
    ``(K, 2)`` int64 index pairs, in the order of ``descriptors1``."""

def __getattr__(name: str) -> Any: ...
