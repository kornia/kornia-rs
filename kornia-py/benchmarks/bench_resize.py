import timeit
from pathlib import Path

import cv2
from PIL import Image as PILImage
import numpy as np

from kornia_rs.image import Image

image_path = str(Path(__file__).resolve().parents[2] / "tests" / "data" / "dog.jpeg")
img_kornia = Image.load(image_path)
img_np = img_kornia.to_numpy()
img_pil = PILImage.open(image_path)
new_size = (128, 128)  # (width, height)
N = 5000  # number of iterations


def resize_image_opencv(img: np.ndarray, new_size: tuple) -> None:
    return cv2.resize(img, new_size, interpolation=cv2.INTER_LINEAR)

def resize_image_pil(img: PILImage.Image, new_size: tuple) -> None:
    return img.resize(new_size, PILImage.BILINEAR)

def resize_image_kornia(img: Image, new_size: tuple) -> None:
    return img.resize(width=new_size[0], height=new_size[1], interpolation="bilinear")

tests = [
    {
        "name": "OpenCV",
        "stmt": "resize_image_opencv(img, new_size)",
        "setup": "from __main__ import resize_image_opencv, img_np, new_size",
        "globals": {"img": img_np, "new_size": new_size},
    },
    {
        "name": "PIL",
        "stmt": "resize_image_pil(img_pil, new_size)",
        "setup": "from __main__ import resize_image_pil, img_pil, new_size",
        "globals": {"img_pil": img_pil, "new_size": new_size},
    },
    {
        "name": "Kornia",
        "stmt": "resize_image_kornia(img, new_size)",
        "setup": "from __main__ import resize_image_kornia, img_kornia, new_size",
        "globals": {"img": img_kornia, "new_size": new_size},
    },
]

for test in tests:
    timer = timeit.Timer(
        stmt=test["stmt"], setup=test["setup"], globals=test["globals"]
    )
    print(f"{test['name']}: {timer.timeit(N)/ N * 1e3:.2f} ms")
