"""Image IO — fast read/write powered by kornia_rs Rust backend."""

from __future__ import annotations

import struct
import numpy as np
from pathlib import Path

# Map user-facing mode to kornia_rs native mode strings
_MODE_TO_NATIVE = {'RGB': 'rgb', 'RGBA': 'rgba', 'L': 'mono'}


def _png_dimensions(data: bytes) -> tuple:
    """Parse width and height from PNG IHDR chunk."""
    # PNG: 8-byte signature, then IHDR chunk: 4 len + 4 'IHDR' + 4 width + 4 height
    if len(data) < 24:
        raise ValueError("Data too short to be a valid PNG")
    width, height = struct.unpack('>II', data[16:24])
    return (height, width)


def read_image(path: str) -> 'Image':
    """Read an image file. Returns RGB Image.

    Supports JPEG, PNG, TIFF via the Rust backend.
    """
    from kornia_rs._image import Image
    from kornia_rs import kornia_rs as _native

    path = str(path)
    arr = _native.io.read_image(path)
    if isinstance(arr, np.ndarray):
        # Use _wrap to avoid a copy — data is already ours
        return Image._wrap(arr)
    raise RuntimeError(f"Failed to read image: {path}")


def decode_image(data: bytes, mode: str = 'RGB') -> 'Image':
    """Decode encoded image bytes (JPEG or PNG) into an Image.

    Detects format from magic bytes:
    - JPEG: starts with \\xff\\xd8
    - PNG:  starts with \\x89PNG

    Tries jpegturbo first for JPEG, falls back to regular jpeg decoder.
    Zero-copy: wraps the freshly-decoded output directly.

    Args:
        data: Encoded image bytes (JPEG or PNG).
        mode: Expected color mode for decoding (e.g. 'RGB', 'L').
    """
    from kornia_rs._image import Image
    from kornia_rs import kornia_rs as _native

    native_mode = _MODE_TO_NATIVE.get(mode, 'rgb')

    if data[:2] == b'\xff\xd8':
        # JPEG — try jpegturbo first, fall back to standard decoder
        try:
            arr = _native.io.decode_image_jpegturbo(data, native_mode)
        except Exception:
            arr = _native.io.decode_image_jpeg(data)
    elif data[:4] == b'\x89PNG':
        image_shape = _png_dimensions(data)
        arr = _native.io.decode_image_png_u8(data, image_shape, native_mode)
    else:
        raise ValueError("Unsupported image format: magic bytes not recognised as JPEG or PNG")

    if not isinstance(arr, np.ndarray):
        raise RuntimeError("Native decoder did not return a numpy array")
    return Image._wrap(arr)


def write_image(image: 'Image', path: str, quality: int = 95) -> None:
    """Write an image to file. Format detected from extension."""
    from kornia_rs import kornia_rs as _native

    path_obj = Path(path)
    ext = path_obj.suffix.lower()
    data = image.data

    # Ensure uint8 for saving
    if data.dtype == np.float32:
        data = np.clip(data * 255, 0, 255).astype(np.uint8)

    if data.ndim == 3 and data.shape[2] == 3:
        if ext in ('.jpg', '.jpeg'):
            _native.io.write_image_jpeg(path, data, "rgb", quality)
            return
        elif ext == '.png':
            _native.io.write_image_png_u8(path, data, "rgb")
            return
        elif ext in ('.tif', '.tiff'):
            _native.io.write_image_tiff_u8(path, data, "rgb")
            return

    raise ValueError(
        f"Unsupported format '{ext}' or channel count {data.shape[2]}. "
        f"Supported: .jpg, .jpeg, .png, .tif, .tiff with 3-channel RGB."
    )
