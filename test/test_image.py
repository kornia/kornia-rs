from pathlib import Path

import kornia_rs as K

DATA_DIR = 'data/'


def test_read_image_jpeg():
    img_path = Path(DATA_DIR) / 'dog.jpeg'
    image = K.read_image_jpeg(img_path.absolute().__str__)
    assert image.shape == (195, 258, 3)