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
