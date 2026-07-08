"""Type stubs for the ``kornia_rs.image`` submodule (compiled extension)."""

from __future__ import annotations

from typing import Any, Optional, Sequence

import numpy as np

from .cuda import Stream

class ImageSize:
    def __init__(self, width: int, height: int) -> None: ...
    @property
    def width(self) -> int: ...
    @property
    def height(self) -> int: ...

class PixelFormat:
    U8: PixelFormat
    U16: PixelFormat
    F32: PixelFormat

class ColorSpace:
    Rgb: ColorSpace
    Bgr: ColorSpace
    Gray: ColorSpace
    Rgba: ColorSpace
    Bgra: ColorSpace
    Hsv: ColorSpace
    Hls: ColorSpace
    Lab: ColorSpace
    Luv: ColorSpace
    Xyz: ColorSpace
    LinearRgb: ColorSpace
    YCbCr: ColorSpace
    Yuv: ColorSpace

class ImageLayout:
    def __init__(
        self, image_size: ImageSize, channels: int, pixel_format: PixelFormat
    ) -> None: ...
    @property
    def image_size(self) -> ImageSize: ...
    @property
    def channels(self) -> int: ...
    @property
    def pixel_format(self) -> PixelFormat: ...

class Image:
    """An owned image/tensor buffer backed by a numpy ``uint8``/``uint16``/``float32`` array.

    Implements the Python buffer protocol (PEP 3118): ``memoryview(img)`` and
    ``np.asarray(img)`` provide zero-copy access to the underlying data.
    Images constructed from read-only sources (e.g. ``bytes``) expose
    ``memoryview(img).readonly == True``; the ``data`` property honours
    the same read-only flag.
    """

    def __init__(self, data: np.ndarray, mode: Optional[str] = ..., color_space: Optional[ColorSpace] = ...) -> None: ...

    # --- constructors ---
    @staticmethod
    def frombuffer(data: Any, mode: Optional[str] = ...) -> Image: ...
    @staticmethod
    def fromarray(data: np.ndarray, mode: Optional[str] = ...) -> Image: ...
    @staticmethod
    def frombytes(
        data: bytes,
        width: int,
        height: int,
        channels: int = ...,
        mode: Optional[str] = ...,
        dtype: Optional[str] = ...,
    ) -> Image: ...
    @staticmethod
    def from_numpy(
        data: np.ndarray,
        mode: Optional[str] = ...,
        color_space: Optional[ColorSpace] = ...,
        copy: bool = ...,
    ) -> Image:
        """Create an Image from a numpy array. Zero-copy by default (``copy=False``);
        pass ``copy=True`` to own an independent aligned copy."""
        ...
    @staticmethod
    def from_buffer(
        data: Any,
        width: int,
        height: int,
        channels: int = ...,
        dtype: str = ...,
        mode: Optional[str] = ...,
        color_space: Optional[ColorSpace] = ...,
    ) -> Image:
        """Zero-copy ingest of a PEP-3118 buffer (bytearray, memoryview, etc.).
        The Image borrows the caller's buffer and keeps the source alive."""
        ...
    @staticmethod
    def zeros(width: int, height: int, channels: int, dtype: str = ...,
              stream: Optional[Stream] = ...) -> Image:
        """Zero-initialized ``(height, width, channels)`` Image. ``dtype`` is
        ``"uint8"`` or ``"float32"``. ``stream=None`` → host; passing a ``Stream``
        allocates directly on that stream's CUDA device (mirrors Rust
        ``zeros_cuda``). To move existing data to a device, use ``.to_cuda(stream)``."""
        ...
    @staticmethod
    def from_dlpack(obj: Any, copy: bool = False) -> Image:
        """Import a DLPack tensor (numpy >= 1.22, PyTorch, CuPy, cuda-python…),
        **inferring the device**: a CPU tensor becomes a host ``Image`` (always
        zero-copy, producer kept alive), a CUDA tensor becomes a device-resident
        ``Image`` on its own source device (not always ``cuda:0``).

        ``copy`` (default ``False``, matching ``torch``/``numpy``'s DLPack
        convention) only affects the CUDA case: ``False`` aliases the producer's
        buffer zero-copy (read-only — writes would land in the producer's
        memory); ``True`` makes an owned, writable device-to-device copy."""
        ...
    @staticmethod
    def load(path: str) -> Image: ...
    @staticmethod
    def decode(data: bytes, mode: str = ...) -> Image: ...
    @staticmethod
    def open(fp: Any, mode: str = ...) -> Image: ...
    @staticmethod
    def new(mode: str, size: tuple[int, int], color: Any = ...) -> Image: ...

    # --- properties ---
    @property
    def width(self) -> int: ...
    @property
    def height(self) -> int: ...
    @property
    def channels(self) -> int: ...
    @property
    def mode(self) -> str: ...
    @property
    def format(self) -> Optional[str]: ...
    @property
    def size(self) -> tuple[int, int]: ...
    @property
    def shape(self) -> tuple[int, int, int]: ...
    @property
    def dtype(self) -> np.dtype: ...
    @property
    def data(self) -> np.ndarray: ...
    @property
    def nbytes(self) -> int: ...
    @property
    def data_ptr(self) -> int:
        """Host address (int) of the underlying contiguous buffer.

        Stable for the lifetime of this ``Image``; hand it (with ``shape`` /
        ``nbytes``) to an external library for a host→device copy.
        """
        ...

    # --- IO ---
    def save(
        self,
        fp: Any,
        format: Optional[str] = ...,
        quality: int = ...,
        compress_level: Optional[int] = ...,
        subsampling: Optional[int] = ...,
    ) -> None: ...
    def encode(
        self,
        format: str,
        quality: int = ...,
        compress_level: Optional[int] = ...,
        subsampling: Optional[int] = ...,
    ) -> bytes: ...
    def tobytes(self) -> bytes: ...
    def to_numpy(self, copy: bool = True) -> np.ndarray:
        """Return the buffer as a numpy array. ``copy=True`` (default) returns
        an independent deep copy; ``copy=False`` returns a zero-copy view.
        The argument is positional-or-keyword (not keyword-only)."""
        ...
    def numpy(self) -> np.ndarray:
        """Numpy array of the underlying buffer. A host image shares memory
        (zero-copy view). A device (CUDA) image is copied back to host (D2H) and
        the result is **read-only** — it does not share the device buffer, so
        writes would be lost; use ``.cpu()`` for a writable host image."""
        ...
    def copy(self) -> Image:
        """Deep copy. Host-only: raises ``ValueError`` on a device image
        (call ``.cpu()`` first)."""
        ...

    # --- device (CUDA) ---
    @property
    def device(self) -> str:
        """``"cpu"`` or ``"cuda:{id}"``."""
        ...
    def cpu(self, stream: Optional[Stream] = ...) -> Image:
        """Copy to host (CPU) memory. A device image is copied back (D2H);
        a host image returns an owned copy."""
        ...
    def to_cuda(self, stream: Optional[Stream] = ...) -> Image:
        """Copy to CUDA device memory (H2D), returning a device ``Image``.
        A no-op handle share if already on device. Requires the ``cuda`` feature."""
        ...
    @property
    def __cuda_array_interface__(self) -> dict:
        """CUDA Array Interface (v3) for zero-copy sharing with cupy / numba /
        cuda-python. Present on device images only; raises ``AttributeError`` on
        host images."""
        ...

    # --- transforms ---
    # NOTE: every transform below reads/writes pixels on the host and is
    # host-only — it raises ``ValueError`` on a device (CUDA) image. Move the
    # image to the host first with ``.cpu()``. (GPU color ops live in
    # ``kornia_rs.cuda`` and take/return device images.)
    def resize(self, width: int, height: int, interpolation: str = ..., antialias: bool = ...) -> Image: ...
    def resize_normalize_to_tensor(
        self,
        width: int,
        height: int,
        mean: Sequence[float],
        std: Sequence[float],
    ) -> Image:
        """Fused resize + per-channel normalize + HWC→CHW into an owned float32
        tensor ``Image`` (CHW), for any target size. NEON/AVX2-vectorized, single
        pass, copy-free. Use ``.numpy()`` for a view and ``.data_ptr`` for the
        host address to hand to an external library. Output is ``(x/255 - mean)/std``.
        """
        ...
    def flip_horizontal(self) -> Image: ...
    def flip_vertical(self) -> Image: ...
    def crop(
        self,
        x: int | tuple[int, int, int, int],
        y: Optional[int] = ...,
        width: Optional[int] = ...,
        height: Optional[int] = ...,
    ) -> Image: ...
    def gaussian_blur(self, kernel_size: int = ..., sigma: float = ...) -> Image: ...
    def box_blur(self, kernel_size: int = ...) -> Image: ...
    def adjust_brightness(self, factor: float) -> Image: ...
    def adjust_contrast(self, factor: float) -> Image: ...
    def adjust_saturation(self, factor: float) -> Image: ...
    def adjust_hue(self, factor: float) -> Image: ...
    def normalize(
        self, mean: tuple[float, float, float], std: tuple[float, float, float]
    ) -> np.ndarray: ...
    def convert(self, mode: str) -> Image: ...
    def to_grayscale(self) -> Image: ...
    def to_rgb(self) -> Image: ...
    def colormap(self, colormap: str) -> Image: ...
    def rotate(self, angle: float) -> Image: ...

    # --- color-space conversion ---
    @property
    def color_space(self) -> ColorSpace: ...
    def cvt_color(self, to: ColorSpace) -> Image: ...
    def to_float(self) -> Image: ...
    def to_uint8(self) -> Image: ...
    def to_gray(self) -> Image: ...
    def to_hsv(self) -> Image: ...
    def to_lab(self) -> Image: ...
    def to_bgr(self) -> Image: ...

    # --- dunders ---
    def __repr__(self) -> str: ...
    def __array__(self, dtype: Any = ..., copy: Any = ...) -> np.ndarray: ...
    def __len__(self) -> int: ...
    def __eq__(self, other: object) -> bool: ...
    def __hash__(self) -> int: ...
    def __enter__(self) -> Image: ...
    def __exit__(self, *args: Any) -> None: ...
    def __dlpack__(
        self,
        *,
        stream: Any = ...,
        max_version: Optional[tuple[int, int]] = ...,
        dl_device: Any = ...,
        copy: Optional[bool] = ...,
    ) -> Any:
        """Export the image buffer as a DLPack capsule (CPU, zero-copy)."""
        ...
    def __dlpack_device__(self) -> tuple[int, int]:
        """Return the DLPack device tuple: ``(kDLCPU=1, device_id=0)``."""
        ...
