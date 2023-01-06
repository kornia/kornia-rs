from pathlib import Path

import kornia_rs as K
from kornia_rs import Tensor as cvTensor

import torch
import numpy as np

DATA_DIR = Path(__file__).parent / "data"


def test_read_image_jpeg():
    # load an image with libjpeg-turbo
    img_path: Path = DATA_DIR / "dog.jpeg"
    cv_tensor: cvTensor = K.read_image_jpeg(str(img_path.absolute()))
    assert cv_tensor.shape == [195, 258, 3]

    # convert to dlpack to import to torch
    th_tensor = torch.utils.dlpack.from_dlpack(cv_tensor)
    assert th_tensor.shape == (195, 258, 3)

    # convert to dlpack to import to numpy
    np_array = np.from_dlpack(cv_tensor)
    assert np_array.shape == (195, 258, 3)

def test_read_image_rs():
    # load an image with image-rs
    img_path: Path = DATA_DIR / "dog.jpeg"
    cv_tensor: cvTensor = K.read_image_rs(str(img_path.absolute()))
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
    K.write_image_jpeg(str(DATA_DIR/"dog_rs.jpeg"), image_encoded)

    image_decoder = K.ImageDecoder()
    image_size: K.ImageSize = image_decoder.read_header(bytes(image_encoded))
    assert image_size.width == 258
    assert image_size.height == 195

    decoded_tensor: K.cvTensor = image_decoder.decode(bytes(image_encoded))
    decoded_image = np.from_dlpack(decoded_tensor)


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

    decoded_tensor: K.cvTensor = image_decoder.decode(bytes(image_encoded))
    decoded_image = np.from_dlpack(decoded_tensor)

    # with 100% quality 3 pixels error
    assert (decoded_image - image).sum() == 3

    # check write/read from file
    K.write_image_jpeg(str(DATA_DIR/"image.jpeg"), image_encoded)
    read_tensor = K.read_image_jpeg(str(DATA_DIR/"image.jpeg"))
    read_image = np.from_dlpack(read_tensor)

    np.testing.assert_allclose(decoded_image, read_image)
