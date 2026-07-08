"""GPU color conversions and fused camera preprocessing (CUDA).

Data model: device pixels live in the unified :class:`kornia_rs.image.Image`
(create one with ``Image.from_numpy(a).to_cuda()``); the color-conversion
functions here take and return such a device ``Image``. Model input (CHW) becomes a
:class:`Tensor` via :class:`CudaPreprocessor`. Everything exports zero-copy
to torch / cupy / cuda-python via ``__dlpack__`` and ``__cuda_array_interface__``.

Requires an NVIDIA driver at runtime (``libcuda``) plus NVRTC (``libnvrtc``,
from the CUDA toolkit or the ``nvidia-cuda-nvrtc-cu12`` pip package —
``pip install kornia-rs[cuda]``). Without them :func:`is_available` returns
``False`` and the rest of kornia_rs works as usual.
"""

from typing import Any, List, Optional, Tuple

import numpy as np

from .image import Image


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

class Tensor:
    """Device-resident [N, C, H, W] tensor — model input (preprocessor output).

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
        """``"cuda:{id}"`` — the device of the ``CudaPreprocessor`` that produced
        this tensor (see its ``device=`` constructor argument; not always 0)."""
    @property
    def data_ptr(self) -> int:
        """Raw device pointer (int) to the contiguous ``[N, C, H, W]`` buffer.
        Bind it directly to a TensorRT input: ``ctx.set_tensor_address(name,
        t.data_ptr)``. Valid while this ``Tensor`` is alive."""
    @property
    def __cuda_array_interface__(self) -> dict:
        """CUDA Array Interface (v3) for zero-copy sharing with cupy / numba /
        cuda-python. The ``stream`` entry carries the producing stream."""
    def numpy(self) -> np.ndarray:
        """Copy to host as float32 numpy (f16 tensors are widened)."""
    def __dlpack__(self, *, stream: object = None, max_version: object = None,
                   dl_device: object = None, copy: object = None) -> object:
        """DLPack capsule (zero-copy). Consumers negotiating ``max_version >=
        (1, 0)`` get a versioned (``dltensor_versioned``) capsule."""
    def __dlpack_device__(self) -> Tuple[int, int]: ...

class CudaPreprocessor:
    """Fused camera preprocessing: raw frame -> normalized CHW tensor, one kernel.

    Frames are passed as a **flat 1-D ``uint8`` array** of the raw packed bytes
    (not an (H, W, C) image): ``H*W*C`` bytes for ``rgb``/``bgr``, or the packed
    plane layout for ``nv12``. Reshape image arrays with ``.reshape(-1)`` first.

    Example::

        pre = CudaPreprocessor(mode="letterbox", format="nv12", f16=True,
                               mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
        t = pre.run(nv12_bytes, w, h, 640, 640)     # Tensor [1,3,640,640] f16
        torch_t = torch.from_dlpack(t)              # zero-copy
    """

    def __init__(self, mode: str = "letterbox", format: str = "rgb",
                 sampling: str = "bilinear", f16: bool = False,
                 mean: Optional[Tuple[float, float, float]] = None,
                 std: Optional[Tuple[float, float, float]] = None,
                 pad_value: int = 114, device: int = 0) -> None:
        """``device``: CUDA device ordinal to build and run this preprocessor on
        (default 0). All its outputs (``Tensor.device``) live there."""
    def run(self, frame: np.ndarray, width: int, height: int,
            out_height: int, out_width: int,
            stream: Optional[Stream] = None) -> Tensor:
        """Flat 1-D ``uint8`` frame bytes -> ``Tensor`` [1, 3, out_h, out_w].

        ``stream``: optional consumer ``Stream`` (e.g. your TensorRT execution
        stream via ``Stream.from_handle``) to fence the output into, so
        ``execute_async_v3`` on it is ordered after this preprocess with no host
        sync."""
    def run_batch(self, frames: List[np.ndarray], width: int, height: int,
                  out_height: int, out_width: int,
                  stream: Optional[Stream] = None) -> Tensor:
        """N flat ``uint8`` same-sized frames -> [N, 3, out_h, out_w]; dtype follows
        f16 flag. ``stream`` fences the output like :meth:`run`."""
    def alloc_output(self, out_height: int, out_width: int,
                     batch: int = 1) -> Tensor:
        """Allocate a zero-initialized ``[batch, 3, out_h, out_w]`` output tensor
        (dtype follows the ``f16`` flag). Preallocate once and reuse across
        frames with :meth:`run_into` for an allocation-free serving loop."""
    def run_into(self, out: Tensor, frame: np.ndarray, width: int, height: int,
                 stream: Optional[Stream] = None) -> None:
        """Preprocess one frame **into** a preallocated ``out`` ([1, 3, H, W],
        matching dtype) — no per-call allocation. Bind ``out.data_ptr`` to a
        fixed TensorRT input once, then call each frame. The write is async: do
        not read/free ``out`` until the work completes (sync, or pass ``stream``
        and order your consumer after it)."""

# GPU color conversions are no longer exposed here — call them through the
# residency-dispatching ``kornia_rs.imgproc.*`` ops (a device ``Image`` routes to
# the GPU kernel, a numpy array runs on the CPU).
