"""Type stubs for the ``kornia_rs.imgproc`` submodule.

Color ops dispatch on residency: a numpy array (HWC) runs on the CPU and returns
a numpy array; a device ``Image`` runs the GPU kernel and returns a device
``Image``; a host ``Image`` runs on the CPU and returns a host ``Image``. u8 ops
return uint8, the f32 perceptual/cylindrical conversions return float32. Naming
is ``<out>_from_<in>``.
"""

from __future__ import annotations

from typing import Any, Optional, Sequence, Union

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
    ) -> tuple[list[SiftKeypoint], np.ndarray | Any]:
        """Returns ``(keypoints, descriptors)``.

        ``keypoints`` is a list of :class:`SiftKeypoint`, always on the host,
        shaped like what ``cv2.SIFT.detectAndCompute`` returns. ``descriptors`` follows the input's residency: a device
        ``Tensor`` of shape ``(N, 128)`` for a device ``Image``, a numpy
        ``(N, 128)`` for a host one. Feed either straight back to ``match``.

        The device descriptors are a fresh allocation, not a view into the
        detector's scratch — the next call overwrites that, so a view would
        change under a caller holding two frames."""

    def match(
        self,
        descriptors_a: np.ndarray | Any,
        descriptors_b: np.ndarray | Any,
        ratio: float = 0.8,
        cross_check: bool = True,
    ) -> np.ndarray:
        """Match two descriptor blocks from ``detect_and_compute``.

        Detection and matching are separable: detect once, then match against
        several frames, or match descriptors that came from elsewhere.

        Dispatches on the descriptors. Two device ``Tensor`` s match on device
        and never cross the bus; two numpy arrays run the NEON matcher. Mixing
        the two is an error rather than a silent transfer — the transfer is the
        expensive part, and hiding it is how a frame budget disappears.

        ``ratio`` is Lowe's ratio; ``>= 1.0`` disables it. Returns an ``(M, 2)``
        int32 array of indices into the two keypoint lists."""


class SiftKeypoint:
    """A single detected keypoint. ``detect_and_compute`` returns a list of
    these, matching the shape of ``cv2.SIFT.detectAndCompute``.

    ``octave``, ``layer`` and ``xi`` are decoded from OpenCV's packed
    ``KeyPoint.octave`` field; ``packed_octave`` is that field verbatim, for
    comparing against ``cv2``.

    Every field is a plain attribute read, but the list is one Python object per
    keypoint — a few thousand on a 752x480 frame. Bulk numeric work wants a
    column, so build it once
    (``np.fromiter((k.x for k in kp), np.float32, len(kp))``) rather than
    re-walking the list per operation."""

    x: float
    """Column coordinate, in pixels of the input image."""
    y: float
    """Row coordinate, in pixels of the input image."""
    size: float
    """Diameter of the meaningful neighbourhood."""
    angle: float
    """Dominant gradient orientation, in degrees clockwise from +x."""
    response: float
    """Contrast at the interpolated extremum; the ``n_features`` ranking key."""
    octave: int
    """Signed octave index. ``-1`` when ``upsample=True``, OpenCV's own
    ``firstOctave = -1``."""
    layer: int
    """Layer within the octave."""
    xi: float
    """Sub-layer offset in ``[-0.5, 0.5)``, quantised to 1/255 by the packing."""
    packed_octave: int
    """The raw packed field, as ``cv2.KeyPoint.octave`` reports it."""
