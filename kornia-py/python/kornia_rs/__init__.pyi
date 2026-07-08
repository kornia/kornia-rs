"""Top-level type stubs for the ``kornia_rs`` package.

``image`` and ``pipeline`` are precisely typed; the remaining submodules are
permissive (``Any``) for now and can be fleshed out incrementally.
"""

from typing import Any, List, Optional, Tuple, Union

import numpy as np

from . import apriltag as apriltag
from . import augmentations as augmentations
from . import cpu as cpu
from . import depth as depth
from . import features as features
from . import image as image
from . import imgproc as imgproc
from . import io as io
from . import k3d as k3d
from . import pipeline as pipeline
from . import segmentation as segmentation

IMAGENET_MEAN: tuple[float, float, float]
"""Standard ImageNet per-channel mean (RGB), matching torchvision."""
IMAGENET_STD: tuple[float, float, float]
"""Standard ImageNet per-channel std (RGB), matching torchvision."""

class Tensor:
    """An N-D tensor, mirroring ``kornia_tensor::Tensor`` — not CUDA-specific
    in spirit, and residency-aware (host or device, e.g. produced by
    :class:`Preprocessor`'s ``[N, C, H, W]`` model input). Only present on a
    ``cuda``-feature build.

    A device-resident ``Tensor`` shares zero-copy with an inference engine:
    hand :attr:`data_ptr` straight to TensorRT's
    ``context.set_tensor_address(name, ptr)``, or consume it via
    ``__dlpack__`` (torch / cupy) or ``__cuda_array_interface__`` (cupy /
    numba, device-only — raises ``AttributeError`` on a host ``Tensor``).
    """

    @property
    def shape(self) -> Tuple[int, int, int, int]: ...
    @property
    def dtype(self) -> str: ...
    @property
    def device(self) -> str:
        """``"cpu"`` or ``"cuda:{id}"`` depending on residency."""
    @property
    def data_ptr(self) -> int:
        """Raw pointer (int) to the contiguous buffer. Bind a device tensor
        directly to a TensorRT input: ``ctx.set_tensor_address(name, t.data_ptr)``.
        Valid while this ``Tensor`` is alive."""
    @property
    def __cuda_array_interface__(self) -> dict:
        """CUDA Array Interface (v3) for zero-copy sharing with cupy / numba /
        cuda-python. Device-only: raises ``AttributeError`` on a host
        ``Tensor``. The ``stream`` entry carries the producing stream."""
    def numpy(self) -> Any:
        """Copy to a float32 numpy array (f16 tensors are widened); a no-copy
        view on a host tensor, a device-to-host copy otherwise."""
    def __dlpack__(self, *, stream: object = None, max_version: object = None,
                   dl_device: object = None, copy: object = None) -> object:
        """DLPack capsule (zero-copy). Consumers negotiating ``max_version >=
        (1, 0)`` get a versioned (``dltensor_versioned``) capsule."""
    def __dlpack_device__(self) -> Tuple[int, int]: ...

class Preprocessor:
    """Fused resize + normalize + HWC→CHW camera preprocessing, mirroring
    ``kornia_imgproc::preprocess::Preprocessor``. Dispatches internally on
    ``device``: ``device=None`` runs on the CPU (rgb/bgr/rgba/bgra sources
    only), ``device=<ordinal>`` (the default, ``0``) runs the fused CUDA
    kernel and additionally supports gray/nv12/yuyv sources. Only present on
    a ``cuda``-feature build.
    """

    def __init__(
        self,
        mode: str = ...,
        format: str = ...,
        sampling: str = ...,
        f16: bool = ...,
        mean: Optional[Tuple[float, float, float]] = ...,
        std: Optional[Tuple[float, float, float]] = ...,
        pad_value: float = ...,
        device: Optional[int] = ...,
    ) -> None: ...
    def run(
        self,
        frame: Union[np.ndarray, List[np.ndarray]],
        width: int,
        height: int,
        out_height: int,
        out_width: int,
        out: Optional[Tensor] = None,
        stream: object = None,
    ) -> Tensor:
        """Preprocess a single raw frame (1-D uint8 numpy array) or a batch
        (list of them) into a model-input ``Tensor``. Pass ``out=`` (from
        :meth:`alloc_output`) to write into a preallocated buffer instead of
        allocating a fresh one — only valid for a single frame, not a batch."""
    def alloc_output(self, out_height: int, out_width: int, batch: int = ...) -> Tensor:
        """Preallocate a ``[batch, 3, out_height, out_width]`` output buffer
        for reuse across calls via ``run(..., out=...)``."""

__version__: str

def __getattr__(name: str) -> Any: ...
