"""Color space conversions — RGB-first, never BGR."""

from __future__ import annotations

import numpy as np
from kornia_rs import kornia_rs as _native


def to_grayscale(data: np.ndarray) -> np.ndarray:
    """Convert RGB image to grayscale (1 channel)."""
    if data.shape[2] == 1:
        return data.copy()
    if data.dtype == np.uint8 and data.shape[2] == 3:
        return _native.imgproc.gray_from_rgb(data)
    # ITU-R 601 luma
    gray = np.dot(data[..., :3].astype(np.float32), [0.299, 0.587, 0.114])
    return gray.astype(np.uint8)[:, :, np.newaxis]


def to_rgb(data: np.ndarray) -> np.ndarray:
    """Convert grayscale to RGB (3 channels)."""
    if data.shape[2] == 3:
        return data.copy()
    if data.shape[2] == 1:
        if data.dtype == np.uint8:
            return _native.imgproc.rgb_from_gray(data)
        return np.repeat(data, 3, axis=2)
    if data.shape[2] == 4:
        if data.dtype == np.uint8:
            return _native.imgproc.rgb_from_rgba(data)
        return data[:, :, :3].copy()
    raise ValueError(f"Cannot convert {data.shape[2]}-channel image to RGB")


def to_bgr(data: np.ndarray) -> np.ndarray:
    """Convert RGB to BGR."""
    if data.shape[2] != 3:
        raise ValueError("to_bgr requires 3-channel input")
    if data.dtype == np.uint8:
        return _native.imgproc.bgr_from_rgb(data)
    return np.ascontiguousarray(data[:, :, ::-1])


def to_rgba(data: np.ndarray, alpha: int = 255) -> np.ndarray:
    """Convert RGB to RGBA with given alpha value."""
    if data.shape[2] == 4:
        return data.copy()
    if data.shape[2] == 3:
        h, w = data.shape[:2]
        alpha_ch = np.full((h, w, 1), alpha, dtype=data.dtype)
        return np.concatenate([data, alpha_ch], axis=2)
    raise ValueError(f"Cannot convert {data.shape[2]}-channel image to RGBA")


def to_hsv(data: np.ndarray) -> np.ndarray:
    """Convert RGB to HSV. Returns float32 in [0,1] range."""
    if data.shape[2] != 3:
        raise ValueError("to_hsv requires 3-channel RGB input")
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

    return np.stack([h, s, v], axis=2).astype(np.float32)
