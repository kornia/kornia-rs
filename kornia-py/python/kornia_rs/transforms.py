"""Deterministic image transforms — Rust-accelerated with numpy fallbacks.

These are the low-level functions that operate on numpy arrays. The Image
class methods call these internally. Prefer using the Image API directly
to minimize copies.
"""

from __future__ import annotations

import math
import numpy as np
from typing import Tuple

from kornia_rs import kornia_rs as _native


def resize(data: np.ndarray, width: int, height: int, interpolation: str = "bilinear") -> np.ndarray:
    """Resize image to (width, height)."""
    if data.dtype == np.uint8 and data.shape[2] == 3:
        return _native.imgproc.resize(data, (height, width), interpolation)
    # Fallback: nearest neighbor via numpy
    h, w, c = data.shape
    row_indices = (np.arange(height) * h / height).astype(int)
    col_indices = (np.arange(width) * w / width).astype(int)
    return data[np.ix_(row_indices, col_indices)]


def flip_horizontal(data: np.ndarray) -> np.ndarray:
    """Flip image horizontally."""
    if data.dtype == np.uint8 and data.shape[2] == 3:
        return _native.imgproc.horizontal_flip(data)
    return np.ascontiguousarray(data[:, ::-1, :])


def flip_vertical(data: np.ndarray) -> np.ndarray:
    """Flip image vertically."""
    if data.dtype == np.uint8 and data.shape[2] == 3:
        return _native.imgproc.vertical_flip(data)
    return np.ascontiguousarray(data[::-1, :, :])


def crop(data: np.ndarray, x: int, y: int, width: int, height: int) -> np.ndarray:
    """Crop image at (x, y) with given width and height."""
    if data.dtype == np.uint8 and data.shape[2] == 3:
        return _native.imgproc.crop(data, x, y, width, height)
    return np.ascontiguousarray(data[y:y + height, x:x + width, :])


def rotate(data: np.ndarray, angle: float) -> np.ndarray:
    """Rotate image by angle degrees (counter-clockwise)."""
    if data.dtype == np.uint8 and data.shape[2] == 3:
        h, w = data.shape[:2]
        cx, cy = w / 2.0, h / 2.0
        cos_a = math.cos(math.radians(angle))
        sin_a = math.sin(math.radians(angle))
        tx = cx - cos_a * cx + sin_a * cy
        ty = cy - sin_a * cx - cos_a * cy
        m = [cos_a, -sin_a, tx, sin_a, cos_a, ty]
        return _native.imgproc.warp_affine(data, m, (h, w), "bilinear")
    # Fallback: 90-degree multiples only
    k = int(round(angle / 90)) % 4
    if k == 0:
        return data.copy()
    return np.ascontiguousarray(np.rot90(data, k=k, axes=(0, 1)))


def gaussian_blur(data: np.ndarray, kernel_size: int = 3, sigma: float = 1.0) -> np.ndarray:
    """Apply Gaussian blur."""
    if data.dtype == np.uint8 and data.shape[2] == 3:
        return _native.imgproc.gaussian_blur(data, (kernel_size, kernel_size), (sigma, sigma))
    # Numpy fallback
    return _gaussian_blur_numpy(data, kernel_size, sigma)


def adjust_brightness(data: np.ndarray, factor: float) -> np.ndarray:
    """Adjust brightness. factor is additive in [0,1] range (0.1 = 10% brighter)."""
    if data.dtype == np.uint8 and data.shape[2] == 3:
        return _native.imgproc.adjust_brightness(data, factor)
    result = data.astype(np.float32) + factor * 255
    return np.clip(result, 0, 255).astype(np.uint8)


def adjust_contrast(data: np.ndarray, factor: float) -> np.ndarray:
    """Adjust contrast. factor=1.0 is identity, >1 increases contrast."""
    mean = data.mean()
    result = (data.astype(np.float32) - mean) * factor + mean
    return np.clip(result, 0, 255).astype(np.uint8)


def adjust_saturation(data: np.ndarray, factor: float) -> np.ndarray:
    """Adjust saturation. factor=1.0 is identity, 0.0 is grayscale."""
    if data.shape[2] != 3:
        return data.copy()
    gray = np.dot(data[..., :3].astype(np.float32), [0.299, 0.587, 0.114])
    gray = gray[:, :, np.newaxis]
    result = gray + factor * (data.astype(np.float32) - gray)
    return np.clip(result, 0, 255).astype(np.uint8)


def adjust_hue(data: np.ndarray, factor: float) -> np.ndarray:
    """Adjust hue. factor is in [-0.5, 0.5], fraction of hue wheel."""
    if data.shape[2] != 3 or factor == 0:
        return data.copy()
    img_f = data.astype(np.float32) / 255.0
    maxc = img_f.max(axis=2)
    minc = img_f.min(axis=2)
    diff = maxc - minc

    h = np.zeros_like(maxc)
    s = np.zeros_like(maxc)
    v = maxc

    mask = diff > 0
    r_mask = mask & (img_f[..., 0] == maxc)
    g_mask = mask & (img_f[..., 1] == maxc) & ~r_mask
    b_mask = mask & ~r_mask & ~g_mask

    h[r_mask] = ((img_f[..., 1] - img_f[..., 2])[r_mask] / diff[r_mask]) % 6
    h[g_mask] = ((img_f[..., 2] - img_f[..., 0])[g_mask] / diff[g_mask]) + 2
    h[b_mask] = ((img_f[..., 0] - img_f[..., 1])[b_mask] / diff[b_mask]) + 4
    h = h / 6.0

    s[mask] = diff[mask] / maxc[mask]

    # Shift hue
    h = (h + factor) % 1.0

    # HSV -> RGB
    h6 = h * 6.0
    i = np.floor(h6).astype(int) % 6
    f = h6 - np.floor(h6)
    p = v * (1 - s)
    q = v * (1 - s * f)
    t = v * (1 - s * (1 - f))

    result = np.zeros_like(img_f)
    for idx, (r, g, b) in enumerate([(v, t, p), (q, v, p), (p, v, t), (p, q, v), (t, p, v), (v, p, q)]):
        m = (i == idx)
        result[..., 0][m] = r[m]
        result[..., 1][m] = g[m]
        result[..., 2][m] = b[m]

    return np.clip(result * 255, 0, 255).astype(np.uint8)


def normalize(data: np.ndarray, mean: Tuple[float, ...], std: Tuple[float, ...]) -> np.ndarray:
    """Normalize image with mean and std per channel. Returns float32."""
    img_f = data.astype(np.float32) / 255.0
    mean_arr = np.array(mean, dtype=np.float32).reshape(1, 1, -1)
    std_arr = np.array(std, dtype=np.float32).reshape(1, 1, -1)
    return ((img_f - mean_arr) / std_arr).astype(np.float32)


def _gaussian_blur_numpy(data: np.ndarray, kernel_size: int = 3, sigma: float = 1.0) -> np.ndarray:
    """Simple Gaussian blur fallback using numpy convolution."""
    x = np.arange(kernel_size) - kernel_size // 2
    kernel_1d = np.exp(-0.5 * (x / sigma) ** 2)
    kernel_1d = kernel_1d / kernel_1d.sum()

    result = data.astype(np.float32)
    pad = kernel_size // 2

    for c in range(data.shape[2]):
        channel = result[:, :, c]
        padded = np.pad(channel, pad, mode='reflect')
        temp = np.zeros_like(channel)
        for i, k in enumerate(kernel_1d):
            temp += k * padded[:, i:i + channel.shape[1]]
        padded = np.pad(temp, pad, mode='reflect')
        out = np.zeros_like(channel)
        for i, k in enumerate(kernel_1d):
            out += k * padded[i:i + channel.shape[0], :]
        result[:, :, c] = out

    return np.clip(result, 0, 255).astype(np.uint8)
