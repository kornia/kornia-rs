"""Type stubs for the ``kornia_rs.pipeline`` submodule (fused preprocessing)."""

from __future__ import annotations

from typing import Sequence

import numpy as np

def resize_normalize_to_tensor(
    image: np.ndarray,
    new_size: tuple[int, int],
    mean: Sequence[float],
    std: Sequence[float],
) -> np.ndarray:
    """Fused resize (general bilinear, any size) + normalize + HWC→CHW.

    ``image`` is an ``(H, W, 3)`` uint8 array; ``new_size`` is ``(dst_h, dst_w)``.
    Returns a fresh ``(3, dst_h, dst_w)`` float32 NCHW array, ``(x/255-mean)/std``.
    """
    ...

class Preprocessor:
    """Pre-allocated fused resize+normalize+HWC→CHW preprocessor (reuses its
    output buffer across calls)."""

    def __init__(
        self,
        src_size: tuple[int, int],
        dst_size: tuple[int, int],
        mean: Sequence[float],
        std: Sequence[float],
    ) -> None: ...
    def __call__(self, image: np.ndarray) -> np.ndarray: ...
    @property
    def output_shape(self) -> tuple[int, int, int]: ...
    @property
    def input_shape(self) -> tuple[int, int, int]: ...
