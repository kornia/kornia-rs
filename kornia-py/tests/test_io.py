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
    img: np.ndarray = K.io.read_image_jpeg(str(img_path.absolute()), "rgb")

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
    img: np.ndarray = K.io.decode_image_jpeg(bytes(img_data))

    # check the image properties
    assert img.shape == (195, 258, 3)
    assert img.dtype == np.uint8

    img_t = torch.from_numpy(img)
    assert img_t.shape == (195, 258, 3)


def test_decode_image_jpegturbo():
    img_path: Path = DATA_DIR / "dog.jpeg"
    with open(img_path, "rb") as f:
        img_data = f.read()
    img: np.ndarray = K.io.decode_image_jpegturbo(bytes(img_data), "rgb")

    # check the image properties
    assert img.shape == (195, 258, 3)
    assert img.dtype == np.uint8

    img_t = torch.from_numpy(img)
    assert img_t.shape == (195, 258, 3)


def test_decode_image_png_u8():
    img_path: Path = DATA_DIR / "dog.png"
    with open(img_path, "rb") as f:
        img_data = f.read()
    img: np.ndarray = K.io.decode_image_png_u8(bytes(img_data), (195, 258), "rgb")

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
    img: np.ndarray = K.io.decode_image_png_u16(bytes(img_data), (32, 32), "rgb")
    # check the image properties
    assert img.shape == (32, 32, 3)
    assert img.dtype == np.uint16

    img_t = torch.from_numpy(img)
    assert img_t.shape == (32, 32, 3)


def test_read_image_png_grayscale():
    """Test reading grayscale PNG image"""
    png_path: Path = DATA_DIR / "dog.png"
    if png_path.exists():
        img_png: np.ndarray = K.io.read_image(str(png_path.absolute()))
        assert img_png.dtype == np.uint8
        assert len(img_png.shape) == 3
        assert img_png.shape[2] == 1
        assert img_png.shape[:2] == (195, 258)


def test_read_image_png_rgb():
    """Test reading RGB PNG image"""
    png_path: Path = DATA_DIR / "dog-rgb8.png"
    if png_path.exists():
        img_png: np.ndarray = K.io.read_image(str(png_path.absolute()))
        assert img_png.dtype == np.uint8
        assert len(img_png.shape) == 3
        assert img_png.shape[2] == 3
        assert img_png.shape[:2] == (195, 258)


def test_read_image():
    """Test the new read_image function with auto-detection"""
    # Test JPEG
    jpeg_path: Path = DATA_DIR / "dog.jpeg"
    img_jpeg: np.ndarray = K.io.read_image(str(jpeg_path.absolute()))
    assert img_jpeg.shape == (195, 258, 3)
    assert img_jpeg.dtype == np.uint8

    # Test PNG uint16 if available
    png16_path: Path = DATA_DIR / "rgb16.png"
    if png16_path.exists():
        img_png16: np.ndarray = K.io.read_image(str(png16_path.absolute()))
        assert img_png16.dtype == np.uint16
        assert img_png16.shape == (32, 32, 3)
        # Test reading an 8-bit rgb tiff
    tiff8_path: Path = DATA_DIR / "dog.tiff"
    if tiff8_path.exists():
        img_tiff8: np.ndarray = K.io.read_image(str(tiff8_path.absolute()))
        assert img_tiff8.dtype == np.uint8
        assert img_tiff8.shape == (195, 258, 3)
        # Test reading an 16-bit rgb tiff
    tiff16_path: Path = DATA_DIR / "rgb16.tiff"
    if tiff16_path.exists():
        img_tiff16: np.ndarray = K.io.read_image(str(tiff16_path.absolute()))
        assert img_tiff16.dtype == np.uint16
        assert img_tiff16.shape == (32, 32, 3)
        # Test reading an 32-bit float rgb tiff
    tiff32_path: Path = DATA_DIR / "rgb32.tiff"
    if tiff32_path.exists():
        img_tiff32: np.ndarray = K.io.read_image(str(tiff32_path.absolute()))
        assert img_tiff32.dtype == np.float32
        assert img_tiff32.shape == (32, 32, 3)


def test_decompress():
    # load an image with libjpeg-turbo
    img_path: Path = DATA_DIR / "dog.jpeg"
    img: np.ndarray = K.io.read_image(str(img_path))

    image_encoder = K.io.ImageEncoder()
    image_encoded: list[int] = image_encoder.encode(img)

    image_decoder = K.io.ImageDecoder()
    image_size: K.image.ImageSize = image_decoder.read_header(bytes(image_encoded))
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

    image_encoder = K.io.ImageEncoder()
    image_encoder.set_quality(100)

    image_encoded: list = image_encoder.encode(img)

    image_decoder = K.io.ImageDecoder()
    image_size: K.image.ImageSize = image_decoder.read_header(bytes(image_encoded))
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
        fcn_write=K.io.write_image_jpeg,
        fcn_read=K.io.read_image_jpeg,
        mode="rgb",
        quality=100,
    )


def test_write_read_jpeg_mono8():
    _test_write_read_impl(
        dtype=np.uint8,
        channels=1,
        file_name="test_write_read_jpeg.jpg",
        fcn_write=K.io.write_image_jpeg,
        fcn_read=K.io.read_image_jpeg,
        mode="mono",
        quality=100,
    )


def test_encode_image_jpeg():
    # Load test image
    img_path: Path = DATA_DIR / "dog.jpeg"
    img: np.ndarray = K.io.read_image_jpeg(str(img_path.absolute()), "rgb")

    # Encode to JPEG bytes
    jpeg_bytes: bytes = K.io.encode_image_jpeg(img, quality=95)

    # Verify it's valid JPEG (magic bytes 0xFF 0xD8)
    assert len(jpeg_bytes) > 2
    assert jpeg_bytes[0] == 0xFF
    assert jpeg_bytes[1] == 0xD8

    # Verify JPEG end marker (0xFF 0xD9)
    assert jpeg_bytes[-2] == 0xFF
    assert jpeg_bytes[-1] == 0xD9

    # Verify we can decode it back
    decoded_img: np.ndarray = K.io.decode_image_jpeg(jpeg_bytes)
    assert decoded_img.shape == (195, 258, 3)
    assert decoded_img.dtype == np.uint8


def test_write_read_jpegturbo():
    _test_write_read_impl(
        dtype=np.uint8,
        channels=3,
        file_name="test_write_read_jpegturbo.jpg",
        fcn_write=K.io.write_image_jpegturbo,
        fcn_read=K.io.read_image_jpegturbo,
        quality=100,
    )


def test_write_read_tiff_rgb8():
    _test_write_read_impl(
        dtype=np.uint8,
        channels=3,
        mode="rgb",
        file_name="test_write_read_tiff.tiff",
        fcn_write=K.io.write_image_tiff_u8,
        fcn_read=K.io.read_image_tiff_u8,
    )


def test_write_read_tiff_mono8():
    _test_write_read_impl(
        dtype=np.uint8,
        channels=1,
        mode="mono",
        file_name="test_write_read_tiff.tiff",
        fcn_write=K.io.write_image_tiff_u8,
        fcn_read=K.io.read_image_tiff_u8,
    )


def test_write_read_tiff_rgb16():
    _test_write_read_impl(
        dtype=np.uint16,
        channels=3,
        mode="rgb",
        file_name="test_write_read_tiff.tiff",
        fcn_write=K.io.write_image_tiff_u16,
        fcn_read=K.io.read_image_tiff_u16,
    )


def test_write_read_tiff_mono16():
    _test_write_read_impl(
        dtype=np.uint16,
        channels=1,
        mode="mono",
        file_name="test_write_read_tiff.tiff",
        fcn_write=K.io.write_image_tiff_u16,
        fcn_read=K.io.read_image_tiff_u16,
    )


def test_write_read_tiff_rgbf32():
    _test_write_read_impl(
        dtype=np.float32,
        channels=3,
        mode="rgb",
        file_name="test_write_read_tiff.tiff",
        fcn_write=K.io.write_image_tiff_f32,
        fcn_read=K.io.read_image_tiff_f32,
    )


def test_write_read_tiff_monof32():
    _test_write_read_impl(
        dtype=np.float32,
        channels=1,
        mode="mono",
        file_name="test_write_read_tiff.tiff",
        fcn_write=K.io.write_image_tiff_f32,
        fcn_read=K.io.read_image_tiff_f32,
    )


def test_write_read_png_rgb8():
    _test_write_read_impl(
        dtype=np.uint8,
        channels=3,
        mode="rgb",
        file_name="test_write_read_png.png",
        fcn_write=K.io.write_image_png_u8,
        fcn_read=K.io.read_image_png_u8,
    )


def test_write_read_png_rgba8():
    _test_write_read_impl(
        dtype=np.uint8,
        channels=4,
        mode="rgba",
        file_name="test_write_read_png.png",
        fcn_write=K.io.write_image_png_u8,
        fcn_read=K.io.read_image_png_u8,
    )


def test_write_read_png_mono8():
    _test_write_read_impl(
        dtype=np.uint8,
        channels=1,
        mode="mono",
        file_name="test_write_read_png.png",
        fcn_write=K.io.write_image_png_u8,
        fcn_read=K.io.read_image_png_u8,
    )


def test_write_read_png_mono16():
    _test_write_read_impl(
        dtype=np.uint16,
        channels=1,
        mode="mono",
        file_name="test_write_read_png.png",
        fcn_write=K.io.write_image_png_u16,
        fcn_read=K.io.read_image_png_u16,
    )


def test_write_read_png_rgb16():
    _test_write_read_impl(
        dtype=np.uint16,
        channels=3,
        mode="rgb",
        file_name="test_write_read_png.png",
        fcn_write=K.io.write_image_png_u16,
        fcn_read=K.io.read_image_png_u16,
    )


def test_write_read_png_rgba16():
    _test_write_read_impl(
        dtype=np.uint16,
        channels=4,
        mode="rgba",
        file_name="test_write_read_png.png",
        fcn_write=K.io.write_image_png_u16,
        fcn_read=K.io.read_image_png_u16,
    )
