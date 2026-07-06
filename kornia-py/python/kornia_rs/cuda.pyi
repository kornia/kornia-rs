"""GPU color conversions and fused camera preprocessing (CUDA).

Data model: pixels stay a :class:`CudaImage` (HWC, typed channels); they
become a :class:`CudaTensor` only as model input (the preprocessor's CHW
output). Both export zero-copy to torch/cupy via ``__dlpack__``.

Requires an NVIDIA driver at runtime (``libcuda``) plus NVRTC (``libnvrtc``,
from the CUDA toolkit or the ``nvidia-cuda-nvrtc-cu12`` pip package —
``pip install kornia-rs[cuda]``). Without them :func:`is_available` returns
``False`` and the rest of kornia_rs works as usual.
"""

from typing import Any, List, Optional, Tuple

import numpy as np

IMAGENET_MEAN: Tuple[float, float, float]
"""Re-export of ``kornia_rs.IMAGENET_MEAN``."""
IMAGENET_STD: Tuple[float, float, float]
"""Re-export of ``kornia_rs.IMAGENET_STD``."""

def is_available() -> bool:
    """True if a CUDA driver and device 0 are usable in this process."""

def upload(array: np.ndarray) -> CudaImage:
    """Upload an (H, W, C) uint8 (C in 1/3/4) or float32 (C in 1/3) array."""

def from_dlpack(tensor: Any, copy: bool = True) -> CudaImage:
    """Import a device-resident (H, W, C) DLPack tensor (torch/cupy).

    copy=True: pixels are copied device-to-device into an owned buffer; the
    producer may free its tensor immediately after.
    copy=False: zero-copy — the image aliases the producer's memory and keeps
    the producer object alive; mutating the producer mutates the image.
    """

class CudaImage:
    """Device-resident image (HWC) — the device twin of ``kornia_rs.image.Image``."""

    @property
    def width(self) -> int: ...
    @property
    def height(self) -> int: ...
    @property
    def channels(self) -> int: ...
    @property
    def dtype(self) -> str: ...
    def download(self) -> np.ndarray:
        """Copy back to host as an (H, W, C) numpy array."""
    def __dlpack__(self, *, stream: Any = None, max_version: Any = None,
                   dl_device: Any = None, copy: Any = None) -> Any: ...
    def __dlpack_device__(self) -> Tuple[int, int]: ...

class CudaTensor:
    """Device-resident [N, C, H, W] tensor — model input (preprocessor output)."""

    @property
    def shape(self) -> Tuple[int, int, int, int]: ...
    @property
    def dtype(self) -> str: ...
    def download(self) -> np.ndarray:
        """Copy to host as float32 numpy (f16 tensors are widened)."""
    def __dlpack__(self, *, stream: Any = None, max_version: Any = None,
                   dl_device: Any = None, copy: Any = None) -> Any: ...
    def __dlpack_device__(self) -> Tuple[int, int]: ...

class CudaPreprocessor:
    """Fused camera preprocessing: raw frame -> normalized CHW tensor, one kernel.

    Frames are passed as a **flat 1-D ``uint8`` array** of the raw packed bytes
    (not an (H, W, C) image): ``H*W*C`` bytes for ``rgb``/``bgr``, or the packed
    plane layout for ``nv12``. Reshape image arrays with ``.reshape(-1)`` first.

    Example::

        pre = CudaPreprocessor(mode="letterbox", format="nv12", f16=True,
                               mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
        t = pre.run(nv12_bytes, w, h, 640, 640)     # CudaTensor [1,3,640,640] f16
        torch_t = torch.from_dlpack(t)              # zero-copy

        # from an (H, W, 3) RGB image array:
        pre = CudaPreprocessor(format="rgb")
        t = pre.run(rgb.reshape(-1), w, h, 224, 224)
    """

    def __init__(self, mode: str = "letterbox", format: str = "rgb",
                 sampling: str = "bilinear", f16: bool = False,
                 mean: Optional[Tuple[float, float, float]] = None,
                 std: Optional[Tuple[float, float, float]] = None,
                 pad_value: int = 114) -> None: ...
    def run(self, frame: np.ndarray, width: int, height: int,
            out_height: int, out_width: int) -> CudaTensor:
        """Flat 1-D ``uint8`` frame bytes -> ``CudaTensor`` [1, 3, out_h, out_w]."""
    def run_batch(self, frames: List[np.ndarray], width: int, height: int,
                  out_height: int, out_width: int) -> CudaTensor:
        """N flat ``uint8`` same-sized frames -> [N, 3, out_h, out_w]; dtype follows f16 flag."""

def gray_from_rgb(img: CudaImage) -> CudaImage: ...
def rgb_from_gray(img: CudaImage) -> CudaImage: ...
def bgr_from_rgb(img: CudaImage) -> CudaImage: ...
def rgba_from_rgb(img: CudaImage) -> CudaImage: ...
def rgb_from_rgba(img: CudaImage) -> CudaImage: ...
def rgb_from_bgra(img: CudaImage) -> CudaImage: ...
def ycbcr_from_rgb(img: CudaImage) -> CudaImage: ...
def rgb_from_ycbcr(img: CudaImage) -> CudaImage: ...
def hsv_from_rgb(img: CudaImage) -> CudaImage: ...
def rgb_from_hsv(img: CudaImage) -> CudaImage: ...
def lab_from_rgb(img: CudaImage) -> CudaImage: ...
def rgb_from_lab(img: CudaImage) -> CudaImage: ...
def sepia_from_rgb(img: CudaImage) -> CudaImage: ...
def apply_colormap(img: CudaImage, colormap: str) -> CudaImage: ...
def rgb_from_bayer(img: CudaImage, pattern: str) -> CudaImage: ...
