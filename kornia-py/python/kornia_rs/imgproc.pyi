"""Type stubs for the ``kornia_rs.imgproc`` submodule.

Color ops dispatch on residency: a numpy array (HWC) runs on the CPU and returns
a numpy array; a device ``Image`` runs the GPU kernel and returns a device
``Image``; a host ``Image`` runs on the CPU and returns a host ``Image``. u8 ops
return uint8, the f32 perceptual/cylindrical conversions return float32. Naming
is ``<out>_from_<in>``.
"""

from __future__ import annotations

from typing import Optional, Sequence, Union

import numpy as np

from .image import Image

# A color op accepts a numpy array or an ``Image`` and returns the same kind
# (numpy -> numpy, Image -> Image; device Images run on the GPU).
_Color = Union[np.ndarray, Image]

# --- channel / colorspace (u8) ---
def rgb_from_gray(image: _Color) -> _Color: ...
def gray_from_rgb(image: _Color) -> _Color: ...
def gray_from_rgb_f32(image: np.ndarray) -> np.ndarray: ...
def bgr_from_rgb(image: _Color) -> _Color: ...
def rgba_from_rgb(image: _Color) -> _Color: ...
def rgb_from_rgba(
    image: _Color, background: Optional[Sequence[int]] = ...
) -> _Color: ...
def rgb_from_bgra(
    image: _Color, background: Optional[Sequence[int]] = ...
) -> _Color: ...
def apply_colormap(image: _Color, colormap: str) -> _Color: ...

# --- f32 perceptual / cylindrical / matrix conversions (3->3) ---
# hsv/lab/ycbcr/sepia have GPU kernels; the rest are CPU-only (a device Image
# raises "no GPU kernel").
def hsv_from_rgb(image: _Color) -> _Color: ...
def rgb_from_hsv(image: _Color) -> _Color: ...
def hls_from_rgb(image: _Color) -> _Color: ...
def rgb_from_hls(image: _Color) -> _Color: ...
def xyz_from_rgb(image: _Color) -> _Color: ...
def rgb_from_xyz(image: _Color) -> _Color: ...
def lab_from_rgb(image: _Color) -> _Color: ...
def rgb_from_lab(image: _Color) -> _Color: ...
def luv_from_rgb(image: _Color) -> _Color: ...
def rgb_from_luv(image: _Color) -> _Color: ...
def linear_rgb_from_rgb(image: _Color) -> _Color: ...
def rgb_from_linear_rgb(image: _Color) -> _Color: ...
def ycbcr_from_rgb(image: _Color) -> _Color: ...
def rgb_from_ycbcr(image: _Color) -> _Color: ...
def yuv_from_rgb(image: _Color) -> _Color: ...
def rgb_from_yuv(image: _Color) -> _Color: ...
def sepia_from_rgb(image: _Color) -> _Color: ...

# --- Bayer demosaic (u8 mosaic -> RGB) ---
def rgb_from_bayer(image: _Color, pattern: str) -> _Color:
    """``pattern`` is the sensor layout: ``"rggb"`` / ``"bggr"`` / ``"grbg"`` / ``"gbrg"``."""
    ...

# --- packed/planar YUV video decode (1-D uint8 buffer -> (H, W, 3) RGB) ---
def rgb_from_yuyv(data: np.ndarray, width: int, height: int) -> np.ndarray: ...
def rgb_from_uyvy(data: np.ndarray, width: int, height: int) -> np.ndarray: ...
def rgb_from_yvyu(data: np.ndarray, width: int, height: int) -> np.ndarray: ...
def rgb_from_nv12(data: np.ndarray, width: int, height: int) -> np.ndarray: ...
def rgb_from_nv21(data: np.ndarray, width: int, height: int) -> np.ndarray: ...
def rgb_from_i420(data: np.ndarray, width: int, height: int) -> np.ndarray: ...
def rgb_from_yv12(data: np.ndarray, width: int, height: int) -> np.ndarray: ...

# --- reverse YUV encode (RGB -> packed/planar YUV, 1-D uint8 buffer) ---
def yuyv_from_rgb(image: np.ndarray) -> np.ndarray:
    """Encode (H,W,3) uint8 RGB to a packed 4:2:2 YUYV buffer (1-D, len W*H*2)."""
    ...
def nv12_from_rgb(image: np.ndarray) -> np.ndarray:
    """Encode (H,W,3) uint8 RGB to a planar 4:2:0 NV12 buffer (1-D, len W*H*3/2)."""
    ...

# --- geometric ---
def resize(
    image: np.ndarray | Image,
    new_size: tuple[int, int],
    interpolation: str,
    antialias: bool = ...,
    out: Image | None = ...,
) -> np.ndarray | Image:
    """``new_size`` is ``(height, width)``; ``interpolation`` is e.g. ``"bilinear"`` / ``"nearest"``.

    A device ``Image`` (f32 3-channel, or u8 1/3/4-channel) runs on the GPU,
    bit-identical to the matching CPU path, and returns a device ``Image``.
    ``antialias`` shapes the u8 bicubic/lanczos kernels (CPU and GPU alike)
    and is ignored for f32 and for nearest/bilinear."""
    ...
def warp_affine(
    image: np.ndarray | Image,
    m: Sequence[float],
    new_size: tuple[int, int],
    interpolation: str,
    out: Optional[np.ndarray] = ...,
) -> np.ndarray | Image:
    """``m`` is the 2x3 affine matrix (row-major, length 6).

    A device ``Image`` (f32, 3-channel) runs on the GPU and returns a device
    ``Image``; ``out=`` is unsupported there."""
    ...
def warp_perspective(
    image: np.ndarray | Image,
    m: Sequence[float],
    new_size: tuple[int, int],
    interpolation: str,
    out: Optional[np.ndarray] = ...,
) -> np.ndarray | Image:
    """``m`` is the 3x3 perspective matrix (row-major, length 9).

    A device ``Image`` (f32, 3-channel) runs on the GPU and returns a device
    ``Image``; ``out=`` is unsupported there."""
    ...
def crop(image: np.ndarray, x: int, y: int, width: int, height: int) -> np.ndarray: ...
def horizontal_flip(image: np.ndarray) -> np.ndarray: ...
def vertical_flip(image: np.ndarray) -> np.ndarray: ...

# --- filtering / enhance / stats ---
def gaussian_blur(
    image: np.ndarray, kernel_size: tuple[int, int], sigma: tuple[float, float]
) -> np.ndarray: ...
def box_blur(image: np.ndarray, kernel_size: tuple[int, int]) -> np.ndarray: ...
def dilate(
    image: np.ndarray | Image,
    kernel: str = ...,
    size: tuple[int, int] = ...,
    border: str = ...,
    constant_value: int = ...,
) -> np.ndarray | Image:
    """Neighborhood maximum over a ``"box"`` / ``"cross"`` / ``"ellipse"``
    structuring element of ``size`` ``(height, width)``. ``border`` is one of
    ``"constant"`` / ``"replicate"`` / ``"reflect101"`` / ``"reflect"`` /
    ``"wrap"``. A u8 device ``Image`` (1/3/4-channel) runs on the GPU,
    bit-identical to the numpy CPU path."""
    ...
def erode(
    image: np.ndarray | Image,
    kernel: str = ...,
    size: tuple[int, int] = ...,
    border: str = ...,
    constant_value: int = ...,
) -> np.ndarray | Image:
    """Neighborhood minimum — see :func:`dilate` for parameters and the GPU path."""
    ...
def adjust_brightness(image: np.ndarray, factor: float) -> np.ndarray: ...
def add_weighted(
    src1: np.ndarray, alpha: float, src2: np.ndarray, beta: float, gamma: float
) -> np.ndarray: ...
def normalize_mean_std(
    image: np.ndarray, mean: Sequence[float], std: Sequence[float]
) -> np.ndarray:
    """Per-channel ``(x/255 - mean) / std`` (3-channel u8 -> float32 HWC)."""
    ...
def compute_histogram(image: np.ndarray | Image, num_bins: int) -> list[int]: ...
def connected_components(
    image: np.ndarray | Image,
    connectivity: int = 8,
) -> tuple[int, np.ndarray | Image]: ...
def canny(
    image: np.ndarray | Image,
    low_threshold: float = 50.0,
    high_threshold: float = 150.0,
    l2_gradient: bool = False,
) -> np.ndarray | Image: ...
def median_blur(
    image: np.ndarray | Image,
    kernel_size: int = 3,
    out: Optional[np.ndarray] = None,
) -> np.ndarray | Image: ...
def bilateral_filter(
    image: np.ndarray | Image,
    d: int = 5,
    sigma_color: float = 50.0,
    sigma_space: float = 50.0,
    out: Optional[np.ndarray] = None,
) -> np.ndarray | Image: ...
def equalize_hist(image: np.ndarray | Image) -> np.ndarray | Image: ...
def clahe(
    image: np.ndarray | Image,
    clip_limit: float = 40.0,
    grid_size: tuple[int, int] = (8, 8),
    out: Optional[np.ndarray] = None,
) -> np.ndarray | Image: ...

class Sift:
    """SIFT detector/descriptor, shaped like ``cv2.SIFT``.

    Dispatches on residency: a device ``Image`` runs the CUDA pipeline, a host
    ``Image`` or a numpy array runs the NEON one. Both reproduce ``cv::SIFT``
    bit for bit and return the same descriptors, so residency changes the speed
    and nothing else — with one exception, ``max_keypoints``, which bounds the
    device buffers only. See its note below.

    The input must be single-channel float32 with values in **0..255** — the
    reference's own internal representation. Normalising to 0..1 changes what
    ``contrast_threshold`` means and silently returns far fewer keypoints.

    The instance owns both backends' scratch, so keep it alive across frames;
    constructing one per call gives up that reuse.
    """

    def __init__(
        self,
        n_features: int = 0,
        n_octave_layers: int = 3,
        contrast_threshold: float = 0.04,
        edge_threshold: float = 10.0,
        sigma: float = 1.6,
        max_keypoints: int = 8192,
        upsample: bool = True,
        max_octaves: int = 0,
        fast_descriptor: bool = False,
    ) -> None:
        """``upsample`` selects ``first_octave``: True is OpenCV's ``-1``, which
        doubles the base image. ``max_octaves=0`` means unlimited.
        ``fast_descriptor`` trades bit-exactness for speed on the GPU.

        ``max_keypoints`` sizes the CUDA plan's device buffers and the device
        path truncates to it; the host path has no ceiling. It is therefore the
        one parameter under which the two backends disagree. For a real keypoint
        budget use ``n_features`` — the reference's ``retainBest``, applied
        identically by both."""

    def detect_and_compute(
        self, image: np.ndarray | Image
    ) -> tuple[np.ndarray, np.ndarray]:
        """Returns ``(keypoints, descriptors)``: ``(N, 6)`` of
        ``x, y, size, angle, response, octave`` and ``(N, 128)``."""

    def match(
        self,
        image_a: np.ndarray | Image,
        image_b: np.ndarray | Image,
        ratio: float = 0.8,
        cross_check: bool = True,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Detect in both images and match.

        Two device ``Image`` s match on device and the descriptors never cross
        the bus; anything else detects and matches on the CPU. Mixing residency
        is an error rather than a silent transfer.

        ``ratio`` is Lowe's ratio; ``>= 1.0`` disables it. Returns
        ``(keypoints_a, keypoints_b, matches)`` with ``matches`` an ``(M, 2)``
        int32 array of indices into the two keypoint arrays."""
