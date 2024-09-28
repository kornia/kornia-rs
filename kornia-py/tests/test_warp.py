from pathlib import Path
import kornia_rs as K

import numpy as np

# TODO: inject this from elsewhere
DATA_DIR = Path(__file__).parents[2] / "tests" / "data"


def test_warp_affine():
    # load an image with libjpeg-turbo
    img_path: Path = DATA_DIR / "dog.jpeg"
    img: np.ndarray = K.read_image_jpeg(str(img_path.absolute()))

    # check the image properties
    assert img.shape == (195, 258, 3)

    affine_matrix = (1.0, 0.0, 0.0, 0.0, 1.0, 0.0)

    img_transformed: np.ndarray = K.warp_affine(
        img, affine_matrix, img.shape[:2], "bilinear"
    )
    assert (img_transformed == img).all()


def test_warp_perspective():
    img_path: Path = DATA_DIR / "dog.jpeg"
    img: np.ndarray = K.read_image_jpeg(str(img_path.absolute()))

    assert img.shape == (195, 258, 3)

    perspective_matrix = (1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0)

    img_transformed: np.ndarray = K.warp_perspective(
        img, perspective_matrix, img.shape[:2], "bilinear"
    )
    assert (img_transformed == img).all()
