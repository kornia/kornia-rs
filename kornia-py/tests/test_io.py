from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Callable

import kornia_rs as K
import torch
import numpy as np

# TODO: inject this from elsewhere
DATA_DIR = Path(__file__).parents[2] / "tests" / "data"


def _test_write_read_impl(
    dtype: np.dtype,
    channels: int,
    file_name: str,
    fcn_write: Callable,
    fcn_read: Callable,
    mode: str,
    quality: int = 100,
) -> None:
    img = np.array(
        [
            [1, 2, 3, 4, 5],
            [6, 7, 8, 9, 10],
            [11, 12, 13, 14, 15],
            [16, 17, 18, 19, 20],
        ],
        dtype=dtype,
    )
    img = np.repeat(img[..., None], channels, axis=-1)

    with tempfile.TemporaryDirectory() as tmpdir:
        img_path = Path(tmpdir) / file_name

        # write: (path, image, mode, quality)
        fcn_write(str(img_path), img, mode, quality)

        # read: (path)
        img_read = fcn_read(str(img_path))

    assert img_read.shape == (4, 5, channels)
    assert np.allclose(img, img_read)


def test_read_image_jpeg():
    img_path: Path = DATA_DIR / "dog.jpeg"
    img: np.ndarray = K.io.read_image(str(img_path.absolute()))

    assert img.shape == (195, 258, 3)
    assert img.dtype == np.uint8

    img_t = torch.from_numpy(img)
    assert img_t.shape == (195, 258, 3)


def test_decode_image_jpeg():
    img_path: Path = DATA_DIR / "dog.jpeg"
    img: np.ndarray = K.io.decode_image(str(img_path))

    assert img.shape == (195, 258, 3)
    assert img.dtype == np.uint8


def test_read_image_png_rgb():
    png_path: Path = DATA_DIR / "dog-rgb8.png"
    if png_path.exists():
        img: np.ndarray = K.io.read_image(str(png_path.absolute()))

        assert img.dtype == np.uint8
        assert img.shape == (195, 258, 3)


def test_write_read_jpeg_rgb8():
    _test_write_read_impl(
        dtype=np.uint8,
        channels=3,
        file_name="test.jpg",
        fcn_write=K.io.write_image,
        fcn_read=K.io.read_image,
        mode="rgb",
        quality=100,
    )


def test_write_read_jpeg_mono8():
    _test_write_read_impl(
        dtype=np.uint8,
        channels=1,
        file_name="test_mono.jpg",
        fcn_write=K.io.write_image,
        fcn_read=K.io.read_image,
        mode="mono",
        quality=100,
    )


def test_write_read_png_rgb8():
    _test_write_read_impl(
        dtype=np.uint8,
        channels=3,
        file_name="test.png",
        fcn_write=K.io.write_image,
        fcn_read=K.io.read_image,
        mode="rgb",
    )


def test_write_read_tiff_rgb16():
    _test_write_read_impl(
        dtype=np.uint16,
        channels=3,
        file_name="test.tiff",
        fcn_write=K.io.write_image,
        fcn_read=K.io.read_image,
        mode="rgb",
    )


def test_write_read_tiff_monof32():
    _test_write_read_impl(
        dtype=np.float32,
        channels=1,
        file_name="test_f32.tiff",
        fcn_write=K.io.write_image,
        fcn_read=K.io.read_image,
        mode="mono",
    )
