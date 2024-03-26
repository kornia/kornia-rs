from pathlib import Path
import kornia_rs as K

import torch
import numpy as np

# TODO: inject this from elsewhere
DATA_DIR = Path(__file__).parents[2] / "tests" / "data"


def test_read_image_jpeg():
    # load an image with libjpeg-turbo
    img_path: Path = DATA_DIR / "dog.jpeg"
    img: np.ndarray = K.read_image_jpeg(str(img_path.absolute()))

    # check the image properties
    assert img.shape == (195, 258, 3)

    img_t = torch.from_numpy(img)
    assert img_t.shape == (195, 258, 3)

# TODO: load other types of images
def test_read_image_any():
    # load an image with image-rs
    img_path: Path = DATA_DIR / "dog.jpeg"
    img: np.ndarray = K.read_image_any(str(img_path.absolute()))

    # check the image properties
    assert img.shape == (195, 258, 3)

    img_t = torch.from_numpy(img)
    assert img_t.shape == (195, 258, 3)

def test_decompress():
    # load an image with libjpeg-turbo
    img_path: Path = DATA_DIR / "dog.jpeg"
    img: np.ndarray = K.read_image_jpeg(str(img_path))

    image_encoder = K.ImageEncoder()
    image_encoded: list[int] = image_encoder.encode(img)

    image_decoder = K.ImageDecoder()
    image_size: K.ImageSize = image_decoder.read_header(bytes(image_encoded))
    assert image_size.width == 258
    assert image_size.height == 195

    decoded_img: np.ndarray = image_decoder.decode(bytes(image_encoded))
    assert decoded_img.shape == (195, 258, 3)


def test_compress_decompress():
    img = np.array([
        [0, 0, 0, 0, 0],
        [0, 127, 255, 127, 0],
        [0, 127, 255, 127, 0],
        [0, 127, 255, 127, 0],
    ], dtype=np.uint8)
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


def test_write_read_jpeg(tmpdir):
    img = np.array([
        [0, 0, 0, 0, 0],
        [0, 127, 255, 127, 0],
        [0, 127, 255, 127, 0],
        [0, 127, 255, 127, 0],
    ], dtype=np.uint8)
    img = np.repeat(img[..., None], 3, axis=-1)

    # write the image to a file
    img_path = tmpdir / "test_write_read_jpeg.jpg"
    K.write_image_jpeg(str(img_path), img)

    # read the image back
    img_read = K.read_image_jpeg(str(img_path))

    # check the image properties
    assert img_read.shape == (4, 5, 3)
    np.allclose(img, img_read)
