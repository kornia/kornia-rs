from pathlib import Path

import kornia_rs as K

DATA_DIR = Path(__file__).parent / "data"


def test_read_image_jpeg():
    img_path: Path = DATA_DIR / "dog.jpeg"
    image = K.read_image_jpeg(str(img_path.absolute()))
    assert image.shape == [195, 258, 3]