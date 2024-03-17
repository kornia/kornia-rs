from pathlib import Path
import kornia_rs as K

import torch
import numpy as np

# TODO: inject this from elsewhere
DATA_DIR = Path(__file__).parents[2] / "tests" / "data"


def test_resize():
    # load an image with libjpeg-turbo
    img_path: Path = DATA_DIR / "dog.jpeg"
    img: np.ndarray = K.read_image_jpeg(str(img_path.absolute()))

    # check the image properties
    assert img.shape == (195, 258, 3)

    img_resized: np.ndarray = K.resize(img, (43, 34), "bilinear")
    assert img_resized.shape == (43, 34, 3)
