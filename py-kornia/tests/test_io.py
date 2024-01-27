from pathlib import Path
import pytest
import random
import asyncio

import kornia_rs as K

import torch
import numpy as np

# TODO: inject this from elsewhere
DATA_DIR = Path(__file__).parents[2] / "tests" / "data"


def test_read_image_jpeg():
    # load an image with libjpeg-turbo
    img_path: Path = DATA_DIR / "dog.jpeg"
    cv_tensor: K.Tensor = K.read_image_jpeg(str(img_path.absolute()))
    assert cv_tensor.shape == [195, 258, 3]

    # convert to dlpack to import to torch
    th_tensor = torch.utils.dlpack.from_dlpack(cv_tensor)
    assert th_tensor.shape == (195, 258, 3)

    # convert to dlpack to import to numpy
    np_array = np.from_dlpack(cv_tensor)
    assert np_array.shape == (195, 258, 3)

# TODO: load other types of images
def test_read_image_any():
    # load an image with image-rs
    img_path: Path = DATA_DIR / "dog.jpeg"
    cv_tensor: K.Tensor = K.read_image_any(str(img_path.absolute()))
    assert cv_tensor.shape == [195, 258, 3]

    # convert to dlpack to import to torch
    th_tensor = torch.utils.dlpack.from_dlpack(cv_tensor)
    assert th_tensor.shape == (195, 258, 3)


def test_decompress():
    # load an image with libjpeg-turbo
    img_path: Path = DATA_DIR / "dog.jpeg"
    tensor_rs = K.read_image_jpeg(str(img_path))
    image = np.from_dlpack(tensor_rs)

    image_encoder = K.ImageEncoder()
    image_encoded: list = image_encoder.encode(image.tobytes(), image.shape)

    image_decoder = K.ImageDecoder()
    image_size: K.ImageSize = image_decoder.read_header(bytes(image_encoded))
    assert image_size.width == 258
    assert image_size.height == 195

    decoded_tensor: K.Tensor = image_decoder.decode(bytes(image_encoded))
    decoded_image = np.from_dlpack(decoded_tensor)
    assert decoded_image.shape == (195, 258, 3)


def test_write_read_jpeg():
    image = np.array([
        [0, 0, 0, 0, 0],
        [0, 127, 255, 127, 0],
        [0, 127, 255, 127, 0],
        [0, 127, 255, 127, 0],
    ], dtype=np.uint8)
    image = np.repeat(image[..., None], 3, axis=-1)

    image_encoder = K.ImageEncoder()
    image_encoder.set_quality(100)

    image_encoded: list = image_encoder.encode(image.tobytes(), image.shape)

    image_decoder = K.ImageDecoder()
    image_size: K.ImageSize = image_decoder.read_header(bytes(image_encoded))
    assert image_size.width == 5
    assert image_size.height == 4

    decoded_tensor: K.Tensor = image_decoder.decode(bytes(image_encoded))
    decoded_image = np.from_dlpack(decoded_tensor)

    # with 100% quality 3 pixels error
    assert (decoded_image - image).sum() == 3
