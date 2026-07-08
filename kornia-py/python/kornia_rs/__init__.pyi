"""Top-level type stubs for the ``kornia_rs`` package.

``image`` and ``pipeline`` are precisely typed; the remaining submodules are
permissive (``Any``) for now and can be fleshed out incrementally.
"""

from typing import Any, Tuple

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
    """An N-D device-resident tensor, mirroring ``kornia_tensor::Tensor`` (not
    CUDA-specific in spirit, though the current binding only backs device
    tensors — e.g. ``kornia_rs.cuda.CudaPreprocessor``'s ``[N, C, H, W]`` model
    input). Only present on a ``cuda``-feature build.

    Share it zero-copy with an inference engine: hand :attr:`data_ptr` straight
    to TensorRT's ``context.set_tensor_address(name, ptr)``, or consume it via
    ``__dlpack__`` (torch / cupy) or ``__cuda_array_interface__`` (cupy / numba).
    """

    @property
    def shape(self) -> Tuple[int, int, int, int]: ...
    @property
    def dtype(self) -> str: ...
    @property
    def device(self) -> str:
        """``"cuda:{id}"`` — the device of the producer (e.g. a
        ``CudaPreprocessor``'s ``device=`` constructor argument; not always 0)."""
    @property
    def data_ptr(self) -> int:
        """Raw device pointer (int) to the contiguous buffer. Bind it directly
        to a TensorRT input: ``ctx.set_tensor_address(name, t.data_ptr)``. Valid
        while this ``Tensor`` is alive."""
    @property
    def __cuda_array_interface__(self) -> dict:
        """CUDA Array Interface (v3) for zero-copy sharing with cupy / numba /
        cuda-python. The ``stream`` entry carries the producing stream."""
    def numpy(self) -> Any:
        """Copy to host as a float32 numpy array (f16 tensors are widened)."""
    def __dlpack__(self, *, stream: object = None, max_version: object = None,
                   dl_device: object = None, copy: object = None) -> object:
        """DLPack capsule (zero-copy). Consumers negotiating ``max_version >=
        (1, 0)`` get a versioned (``dltensor_versioned``) capsule."""
    def __dlpack_device__(self) -> Tuple[int, int]: ...

__version__: str

def __getattr__(name: str) -> Any: ...
