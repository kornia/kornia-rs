"""Core Image class — a fast, PIL-like interface for image processing.

Designed for multiprocess safety (Ray Data, multiprocessing) via serialization
support and numpy-backed data that supports zero-copy sharing.

Copy policy:
  - Image.fromarray   → zero-copy (wraps caller's array)
  - Image.load        → zero-copy (wraps freshly-decoded data)
  - Image.fromcompressed → zero-copy (wraps freshly-decoded data)
  - _wrap(arr)        → zero-copy (internal, for transform results)
  - img.data          → zero-copy (returns backing buffer)
  - img.to_numpy()    → copies (explicit user request)
"""

from __future__ import annotations

import numpy as np
from typing import Optional, Tuple, Union


# Image modes
MODE_L = 'L'        # grayscale, 1 channel
MODE_RGB = 'RGB'    # 3 channels
MODE_RGBA = 'RGBA'  # 4 channels

_CHANNELS_TO_MODE = {1: MODE_L, 3: MODE_RGB, 4: MODE_RGBA}


class Image:
    """A PIL-like image object backed by numpy arrays and Rust operations.

    Always stores data as HWC (height, width, channels) numpy arrays.
    RGB color order by default (never BGR like OpenCV).

    Thread-safe and serialization-friendly for use with Ray Data,
    multiprocessing, and other parallel execution frameworks.
    """

    __slots__ = ('_data', '_mode', '__weakref__')

    def __init__(self, data: np.ndarray, mode: Optional[str] = None):
        """Internal constructor. Use Image.load, Image.fromarray, or Image.fromcompressed."""
        if data.ndim == 2:
            data = data[:, :, np.newaxis]
        if data.ndim != 3:
            raise ValueError(f"Expected 2D or 3D array, got {data.ndim}D")
        self._data = data
        self._mode = mode or _CHANNELS_TO_MODE.get(self._data.shape[2], f'{self._data.shape[2]}ch')

    @classmethod
    def _wrap(cls, data: np.ndarray, mode: Optional[str] = None) -> 'Image':
        """Internal: wrap a freshly-allocated array without copying.

        Only use for data that is already exclusively owned (e.g. output
        from a Rust op or a numpy operation that allocated new memory).
        """
        img = object.__new__(cls)
        img._data = data
        img._mode = mode or _CHANNELS_TO_MODE.get(data.shape[2], f'{data.shape[2]}ch')
        return img

    # --- Serialization support for multiprocess (Ray Data, etc.) ---

    def __reduce__(self):
        """Enable serialization for multiprocess frameworks like Ray."""
        return (Image, (self._data, self._mode))

    def __getstate__(self):
        return (self._data, self._mode)

    def __setstate__(self, state):
        if isinstance(state, tuple):
            self._data, self._mode = state
        else:
            # Backwards compat with old state format
            self._data = state
            self._mode = _CHANNELS_TO_MODE.get(state.shape[2], f'{state.shape[2]}ch')

    # --- Static constructors ---

    @staticmethod
    def load(path: str) -> 'Image':
        """Load an image from file (JPEG, PNG, TIFF). Always returns RGB.

        Zero-copy: wraps the freshly-decoded Rust output directly.
        """
        from kornia_rs._io import read_image
        return read_image(path)

    @staticmethod
    def decode(data: bytes, mode: str = 'RGB') -> 'Image':
        """Decode encoded image bytes (JPEG, PNG) into an Image.

        Zero-copy: wraps the freshly-decoded output directly.

        Args:
            data: Encoded image bytes (JPEG or PNG).
            mode: Expected color mode for decoding.
        """
        from kornia_rs._io import decode_image
        return decode_image(data, mode=mode)

    @staticmethod
    def frombytes(data, width: int = None, height: int = None,
                  channels: int = 3, mode: Optional[str] = None) -> 'Image':
        """Create an Image from raw pixel bytes. Zero-copy.

        Accepts any buffer-protocol object: bytes, bytearray, memoryview,
        or numpy arrays. No numpy import needed by callers.

        Args:
            data: Raw pixel data (buffer-protocol object or numpy array).
            width: Image width. Optional if data has a .shape attribute.
            height: Image height. Optional if data has a .shape attribute.
            channels: Number of channels (default 3). Ignored if data has shape.
            mode: Color mode ('L', 'RGB', 'RGBA'). Auto-detected if None.
        """
        if hasattr(data, 'shape'):
            # numpy array or similar — infer shape, zero-copy wrap
            arr = data if isinstance(data, np.ndarray) else np.asarray(data)
            if arr.ndim == 2:
                arr = arr[:, :, np.newaxis]
            if arr.ndim != 3:
                raise ValueError(f"Expected 2D or 3D array, got {arr.ndim}D")
            return Image._wrap(arr, mode=mode)
        else:
            # Raw bytes/bytearray/memoryview — need explicit dimensions
            if width is None or height is None:
                raise ValueError("width and height are required for raw bytes input")
            arr = np.frombuffer(data, dtype=np.uint8).reshape(height, width, channels)
            return Image._wrap(arr, mode=mode)

    # --- Properties ---

    @property
    def width(self) -> int:
        return self._data.shape[1]

    @property
    def height(self) -> int:
        return self._data.shape[0]

    @property
    def channels(self) -> int:
        return self._data.shape[2]

    @property
    def mode(self) -> str:
        """Color mode: 'L' (grayscale), 'RGB', 'RGBA'."""
        return self._mode

    @property
    def size(self) -> Tuple[int, int]:
        """(width, height) — PIL convention."""
        return (self.width, self.height)

    @property
    def shape(self) -> Tuple[int, int, int]:
        """(height, width, channels)."""
        return self._data.shape

    @property
    def dtype(self) -> np.dtype:
        return self._data.dtype

    @property
    def data(self) -> np.ndarray:
        """The underlying numpy array (HWC). Zero-copy."""
        return self._data

    @property
    def nbytes(self) -> int:
        """Total bytes of image data."""
        return self._data.nbytes

    # --- IO ---

    def save(self, path: str, quality: int = 95) -> None:
        """Save image to file. Format detected from extension."""
        from kornia_rs._io import write_image
        write_image(self, path, quality=quality)

    def to_numpy(self) -> np.ndarray:
        """Return a copy of the underlying numpy array."""
        return self._data.copy()

    def copy(self) -> 'Image':
        """Return a deep copy of this image."""
        return Image._wrap(self._data.copy(), mode=self._mode)

    # --- Chainable transforms ---
    # Each returns a NEW Image via _wrap (zero-copy wrap of fresh data).
    # The Rust ops and numpy ops already allocate new arrays, so no
    # additional copy is needed.

    def resize(self, width: int, height: int, interpolation: str = "bilinear") -> 'Image':
        from kornia_rs.transforms import resize
        return Image._wrap(resize(self._data, width, height, interpolation), self._mode)

    def flip_horizontal(self) -> 'Image':
        from kornia_rs.transforms import flip_horizontal
        return Image._wrap(flip_horizontal(self._data), self._mode)

    def flip_vertical(self) -> 'Image':
        from kornia_rs.transforms import flip_vertical
        return Image._wrap(flip_vertical(self._data), self._mode)

    def crop(self, x: int, y: int, width: int, height: int) -> 'Image':
        from kornia_rs.transforms import crop
        return Image._wrap(crop(self._data, x, y, width, height), self._mode)

    def rotate(self, angle: float) -> 'Image':
        from kornia_rs.transforms import rotate
        return Image._wrap(rotate(self._data, angle), self._mode)

    def to_grayscale(self) -> 'Image':
        from kornia_rs.color import to_grayscale
        return Image._wrap(to_grayscale(self._data), MODE_L)

    def to_rgb(self) -> 'Image':
        from kornia_rs.color import to_rgb
        return Image._wrap(to_rgb(self._data), MODE_RGB)

    def adjust_brightness(self, factor: float) -> 'Image':
        from kornia_rs.transforms import adjust_brightness
        return Image._wrap(adjust_brightness(self._data, factor), self._mode)

    def adjust_contrast(self, factor: float) -> 'Image':
        from kornia_rs.transforms import adjust_contrast
        return Image._wrap(adjust_contrast(self._data, factor), self._mode)

    def adjust_saturation(self, factor: float) -> 'Image':
        from kornia_rs.transforms import adjust_saturation
        return Image._wrap(adjust_saturation(self._data, factor), self._mode)

    def gaussian_blur(self, kernel_size: int = 3, sigma: float = 1.0) -> 'Image':
        from kornia_rs.transforms import gaussian_blur
        return Image._wrap(gaussian_blur(self._data, kernel_size, sigma), self._mode)

    def normalize(self, mean: Tuple[float, ...], std: Tuple[float, ...]) -> 'Image':
        """Normalize image to float32 using mean and std per channel."""
        from kornia_rs.transforms import normalize
        return Image._wrap(normalize(self._data, mean, std), self._mode)

    # --- Context manager ---

    def __enter__(self) -> 'Image':
        return self

    def __exit__(self, *args) -> None:
        pass

    # --- Dunder methods ---

    def __repr__(self) -> str:
        return f"Image(mode={self._mode}, size={self.width}x{self.height}, dtype={self.dtype})"

    def __eq__(self, other) -> bool:
        if not isinstance(other, Image):
            return NotImplemented
        return self._mode == other._mode and np.array_equal(self._data, other._data)

    def __array__(self, dtype=None, copy=None) -> np.ndarray:
        """Support np.array(img) and Ray Data Arrow conversion."""
        if dtype is not None:
            return self._data.astype(dtype, copy=True)
        if copy:
            return self._data.copy()
        return self._data

    def __len__(self) -> int:
        """Number of rows (height). Useful for Ray Data batching."""
        return self._data.shape[0]
