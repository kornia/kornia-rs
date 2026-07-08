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
    :class:`Preprocessor`'s ``[N, C, H, W]`` model input). Always available;
    device residency additionally needs the ``cuda`` feature (a CPU-only build
    produces host tensors).

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
    ``kornia_imgproc::preprocess::Preprocessor``. The ``stream`` selects the
    device, exactly like the rest of the API (``Image.to_cuda(stream)``,
    ``Image.zeros(stream=)``) and mirroring Rust ``build_cuda(stream)``:
    ``stream=None`` (the default) runs on the **CPU** (rgb/bgr/rgba/bgra sources
    only); ``stream=<Stream>`` runs the fused **CUDA** kernel on that stream's
    device and additionally supports gray/nv12/yuyv sources. The GPU path needs
    the ``cuda`` feature; ``stream=None`` (CPU) works on a CPU-only build.
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
        stream: Optional[Any] = ...,
    ) -> None: ...
    def run(
        self,
        frame: Union[np.ndarray, List[np.ndarray], Any],
        width: int,
        height: int,
        out_height: int,
        out_width: int,
        out: Optional[Tensor] = None,
        consumer_stream: object = None,
    ) -> Tensor:
        """Preprocess a single raw frame (1-D uint8 numpy array), a batch (list
        of them), or an already-decoded interleaved ``Image`` (rgb/bgr/rgba/bgra
        — a device ``Image`` feeds the fused kernel its buffer zero-copy) into a
        model-input ``Tensor``. Pass ``out=`` (from :meth:`alloc_output`) to
        write into a preallocated buffer instead of allocating a fresh one — only
        valid for a single raw frame. ``consumer_stream`` (GPU only) fences the
        output into your engine's execution stream."""
    def alloc_output(self, out_height: int, out_width: int, batch: int = ...) -> Tensor:
        """Preallocate a ``[batch, 3, out_height, out_width]`` output buffer
        for reuse across calls via ``run(..., out=...)``."""

__version__: str

def __getattr__(name: str) -> Any: ...
