from __future__ import annotations
import tempfile
from pathlib import Path
from typing import Callable
import kornia_rs as K

import torch
import numpy as np

# TODO: inject this from elsewhere
DATA_DIR = Path(__file__).parents[2] / "tests" / "data"


# use this to test the write and read functions
def _test_write_read_impl(
    dtype: np.dtype,
    channels: int,
    file_name: str,
    fcn_write: Callable,
    fcn_read: Callable,
    mode: str | None = None,
    quality: int | None = None,
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

    # write the image to a file
    with tempfile.TemporaryDirectory() as tmpdir:
        img_path = Path(tmpdir) / file_name
        read_kwargs = {} if mode is None else {"mode": mode}
        write_kwargs = read_kwargs.copy()
        if quality is not None:
            write_kwargs["quality"] = quality
        fcn_write(str(img_path), img, **write_kwargs)

        # read the image back
        img_read = fcn_read(str(img_path), **read_kwargs)

    # check the image properties
    assert img_read.shape == (4, 5, channels)
    assert np.allclose(img, img_read)


def test_read_image_jpeg():
    # load an image with libjpeg-turbo
    img_path: Path = DATA_DIR / "dog.jpeg"
    img: np.ndarray = K.read_image_jpeg(str(img_path.absolute()), "rgb")

    # check the image properties
    assert img.shape == (195, 258, 3)
    assert img.dtype == np.uint8

    img_t = torch.from_numpy(img)
    assert img_t.shape == (195, 258, 3)


def test_decode_image_jpeg():
    # load an image with libjpeg-turbo
    img_path: Path = DATA_DIR / "dog.jpeg"
    with open(img_path, "rb") as f:
        img_data = f.read()
    img: np.ndarray = K.decode_image_jpeg(bytes(img_data), (195, 258), "rgb")

    # check the image properties
    assert img.shape == (195, 258, 3)
    assert img.dtype == np.uint8

    img_t = torch.from_numpy(img)
    assert img_t.shape == (195, 258, 3)


def test_decode_image_jpegturbo():
    img_path: Path = DATA_DIR / "dog.jpeg"
    with open(img_path, "rb") as f:
        img_data = f.read()
    img: np.ndarray = K.decode_image_jpegturbo(bytes(img_data), "rgb")

    # check the image properties
    assert img.shape == (195, 258, 3)
    assert img.dtype == np.uint8

    img_t = torch.from_numpy(img)
    assert img_t.shape == (195, 258, 3)


def test_decode_image_png_u8():
    img_path: Path = DATA_DIR / "dog.png"
    with open(img_path, "rb") as f:
        img_data = f.read()
    img: np.ndarray = K.decode_image_png_u8(bytes(img_data), (195, 258), "rgb")

    # check the image properties
    assert img.shape == (195, 258, 3)
    assert img.dtype == np.uint8

    img_t = torch.from_numpy(img)
    assert img_t.shape == (195, 258, 3)


def test_decode_image_png_u16():
    img_path: Path = DATA_DIR / "rgb16.png"
    with open(img_path, "rb") as f:
        img_data = f.read()
    # img_size: np.ndarray = np.array([32, 32])
    img: np.ndarray = K.decode_image_png_u16(bytes(img_data), (32, 32), "rgb")

    # check the image properties
    assert img.shape == (32, 32, 3)
    assert img.dtype == np.uint16

    img_t = torch.from_numpy(img)
    assert img_t.shape == (32, 32, 3)


def test_read_image_any():
    # load an image with image-rs
    img_path: Path = DATA_DIR / "dog.jpeg"
    img: np.ndarray = K.read_image_any(str(img_path.absolute()))

    # check the image properties
    assert img.shape == (195, 258, 3)
    assert img.dtype == np.uint8
    img_t = torch.from_numpy(img)
    assert img_t.shape == (195, 258, 3)


def test_decompress():
    # load an image with libjpeg-turbo
    img_path: Path = DATA_DIR / "dog.jpeg"
    img: np.ndarray = K.read_image_jpeg(str(img_path), "rgb")

    image_encoder = K.ImageEncoder()
    image_encoded: list[int] = image_encoder.encode(img)

    image_decoder = K.ImageDecoder()
    image_size: K.ImageSize = image_decoder.read_header(bytes(image_encoded))
    assert image_size.width == 258
    assert image_size.height == 195

    decoded_img: np.ndarray = image_decoder.decode(bytes(image_encoded))
    assert decoded_img.shape == (195, 258, 3)


def test_compress_decompress():
    img = np.array(
        [
            [0, 0, 0, 0, 0],
            [0, 127, 255, 127, 0],
            [0, 127, 255, 127, 0],
            [0, 127, 255, 127, 0],
        ],
        dtype=np.uint8,
    )
    img = np.repeat(img[..., None], 3, axis=-1)

    image_encoder = K.ImageEncoder()
    image_encoder.set_quality(100)

    image_encoded: list = image_encoder.encode(img)

    image_decoder = K.ImageDecoder()
    image_size: K.ImageSize = image_decoder.read_header(bytes(image_encoded))
    assert image_size.width == 5
    assert image_size.height == 4

    decoded_img: np.ndarray = image_decoder.decode(bytes(image_encoded))

    # with 100% quality 3 pixels error
    assert (decoded_img - img).sum() == 3


def test_write_read_jpeg_rgb8():
    _test_write_read_impl(
        dtype=np.uint8,
        channels=3,
        file_name="test_write_read_jpeg.jpg",
        fcn_write=K.write_image_jpeg,
        fcn_read=K.read_image_jpeg,
        mode="rgb",
        quality=100,
    )


def test_write_read_jpeg_mono8():
    _test_write_read_impl(
        dtype=np.uint8,
        channels=1,
        file_name="test_write_read_jpeg.jpg",
        fcn_write=K.write_image_jpeg,
        fcn_read=K.read_image_jpeg,
        mode="mono",
        quality=100,
    )


def test_write_read_jpegturbo():
    _test_write_read_impl(
        dtype=np.uint8,
        channels=3,
        file_name="test_write_read_jpegturbo.jpg",
        fcn_write=K.write_image_jpegturbo,
        fcn_read=K.read_image_jpegturbo,
        quality=100,
    )


def test_write_read_tiff_rgb8():
    _test_write_read_impl(
        dtype=np.uint8,
        channels=3,
        mode="rgb",
        file_name="test_write_read_tiff.tiff",
        fcn_write=K.write_image_tiff_u8,
        fcn_read=K.read_image_tiff_u8,
    )


def test_write_read_tiff_mono8():
    _test_write_read_impl(
        dtype=np.uint8,
        channels=1,
        mode="mono",
        file_name="test_write_read_tiff.tiff",
        fcn_write=K.write_image_tiff_u8,
        fcn_read=K.read_image_tiff_u8,
    )


def test_write_read_tiff_rgb16():
    _test_write_read_impl(
        dtype=np.uint16,
        channels=3,
        mode="rgb",
        file_name="test_write_read_tiff.tiff",
        fcn_write=K.write_image_tiff_u16,
        fcn_read=K.read_image_tiff_u16,
    )


def test_write_read_tiff_mono16():
    _test_write_read_impl(
        dtype=np.uint16,
        channels=1,
        mode="mono",
        file_name="test_write_read_tiff.tiff",
        fcn_write=K.write_image_tiff_u16,
        fcn_read=K.read_image_tiff_u16,
    )


def test_write_read_tiff_rgbf32():
    _test_write_read_impl(
        dtype=np.float32,
        channels=3,
        mode="rgb",
        file_name="test_write_read_tiff.tiff",
        fcn_write=K.write_image_tiff_f32,
        fcn_read=K.read_image_tiff_f32,
    )


def test_write_read_tiff_monof32():
    _test_write_read_impl(
        dtype=np.float32,
        channels=1,
        mode="mono",
        file_name="test_write_read_tiff.tiff",
        fcn_write=K.write_image_tiff_f32,
        fcn_read=K.read_image_tiff_f32,
    )


def test_write_read_png_rgb8():
    _test_write_read_impl(
        dtype=np.uint8,
        channels=3,
        mode="rgb",
        file_name="test_write_read_png.png",
        fcn_write=K.write_image_png_u8,
        fcn_read=K.read_image_png_u8,
    )


def test_write_read_png_rgba8():
    _test_write_read_impl(
        dtype=np.uint8,
        channels=4,
        mode="rgba",
        file_name="test_write_read_png.png",
        fcn_write=K.write_image_png_u8,
        fcn_read=K.read_image_png_u8,
    )


def test_write_read_png_mono8():
    _test_write_read_impl(
        dtype=np.uint8,
        channels=1,
        mode="mono",
        file_name="test_write_read_png.png",
        fcn_write=K.write_image_png_u8,
        fcn_read=K.read_image_png_u8,
    )


def test_write_read_png_mono16():
    _test_write_read_impl(
        dtype=np.uint16,
        channels=1,
        mode="mono",
        file_name="test_write_read_png.png",
        fcn_write=K.write_image_png_u16,
        fcn_read=K.read_image_png_u16,
    )


def test_write_read_png_rgb16():
    _test_write_read_impl(
        dtype=np.uint16,
        channels=3,
        mode="rgb",
        file_name="test_write_read_png.png",
        fcn_write=K.write_image_png_u16,
        fcn_read=K.read_image_png_u16,
    )


def test_write_read_png_rgba16():
    _test_write_read_impl(
        dtype=np.uint16,
        channels=4,
        mode="rgba",
        file_name="test_write_read_png.png",
        fcn_write=K.write_image_png_u16,
        fcn_read=K.read_image_png_u16,
    )
