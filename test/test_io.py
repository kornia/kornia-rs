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
    dlpack = K.cvtensor_to_dlpack(cv_tensor)
    th_tensor = torch.utils.dlpack.from_dlpack(dlpack)
    assert th_tensor.shape == (195, 258, 3)

    # TODO: needs to be fixed
    # convert to dlpack to import to numpy
    #dlpack = K.cvtensor_to_dlpack(cv_tensor)
    #np_array = np._from_dlpack(dlpack)
    #assert np_array.shape == (195, 258, 3)

def test_read_image_rs():
    # load an image with image-rs
    img_path: Path = DATA_DIR / "dog.jpeg"
    cv_tensor: cvTensor = K.read_image_rs(str(img_path.absolute()))
    assert cv_tensor.shape == [195, 258, 3]

    # convert to dlpack to import to torch
    dlpack = K.cvtensor_to_dlpack(cv_tensor)
    th_tensor = torch.utils.dlpack.from_dlpack(dlpack)
    assert th_tensor.shape == (195, 258, 3)