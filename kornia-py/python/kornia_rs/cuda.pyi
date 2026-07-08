"""GPU color conversions and fused camera preprocessing (CUDA).

Data model: device pixels live in the unified :class:`kornia_rs.image.Image`
(create one with ``Image.from_numpy(a).to_cuda()``); the color-conversion
functions here take and return such a device ``Image``. Model input (CHW) becomes a
:class:`kornia_rs.Tensor` via :class:`kornia_rs.Preprocessor`. Everything exports zero-copy
to torch / cupy / cuda-python via ``__dlpack__`` and ``__cuda_array_interface__``.

Requires an NVIDIA driver at runtime (``libcuda``) plus NVRTC (``libnvrtc``,
from the CUDA toolkit or the ``nvidia-cuda-nvrtc-cu12`` pip package —
``pip install kornia-rs[cuda]``). Without them :func:`is_available` returns
``False`` and the rest of kornia_rs works as usual.
"""

from typing import Any, Tuple

from . import Tensor as Tensor


class Stream:
    """A CUDA stream handle, shareable with kornia device transfers and the
    DLPack / cuda-python stream protocols. The stream's device is the selector
    for where ``Image.to_cuda(stream)`` / ``Image.zeros(..., stream=stream)``
    place data. Only meaningful on a ``cuda`` build."""

    @staticmethod
    def default(device: int = ...) -> Stream:
        """The process-wide default CUDA stream for ``device`` (default 0)."""
        ...
    @staticmethod
    def from_handle(handle: int) -> Stream:
        """Adopt an existing raw ``CUstream`` handle (an integer) from NVIDIA's
        stack — e.g. ``cuda.core.Stream.handle``, a cuda-python ``CUstream``, or
        a CuPy ``stream.ptr``. kornia does **not** take ownership (the stream is
        never destroyed here); device ops fence their work into it via a CUDA
        event so your later work on the same stream is ordered after kornia's."""
        ...
    @staticmethod
    def from_cuda_stream(obj: Any) -> Stream:
        """Adopt a stream from any object implementing the cuda-python /
        ``cuda.core`` protocol (``__cuda_stream__() -> (version, handle)``),
        exposing an integer ``.ptr`` / ``.handle``, or that is itself an int."""
        ...
    def synchronize(self) -> None:
        """Block the host until all work on this stream completes."""
        ...
    @property
    def cuda_stream_ptr(self) -> int:
        """Raw ``CUstream`` handle as an integer (DLPack ``stream=`` protocol)."""
        ...
    def __cuda_stream__(self) -> tuple[int, int]:
        """cuda-python / ``cuda.core`` protocol: ``(protocol_version, handle)``."""
        ...

IMAGENET_MEAN: Tuple[float, float, float]
"""Re-export of ``kornia_rs.IMAGENET_MEAN``."""
IMAGENET_STD: Tuple[float, float, float]
"""Re-export of ``kornia_rs.IMAGENET_STD``."""

def is_available() -> bool:
    """True if a CUDA driver and device 0 are usable in this process."""

def mem_get_info() -> Tuple[int, int]:
    """Free and total device-0 global memory in bytes, as ``(free, total)``.

    Wraps ``cuMemGetInfo`` (synchronizes the default stream first). Bracket a
    loop with it to assert the free byte count returns to baseline — i.e. no
    device memory leaked across the iterations."""

# GPU color conversions are no longer exposed here — call them through the
# residency-dispatching ``kornia_rs.imgproc.*`` ops (a device ``Image`` routes to
# the GPU kernel, a numpy array runs on the CPU).
