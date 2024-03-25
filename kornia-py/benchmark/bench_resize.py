import timeit

import cv2
from PIL import Image
import kornia_rs
import numpy as np
# import tensorflow as tf

image_path = "tests/data/dog.jpeg"
img = kornia_rs.read_image_jpeg(image_path)
img_pil = Image.open(image_path)
new_size = (128, 128)
N = 5000  # number of iterations


def resize_image_opencv(img: np.ndarray, new_size: tuple) -> None:
    return cv2.resize(img, new_size, interpolation=cv2.INTER_LINEAR)

def resize_image_pil(img: Image.Image, new_size: tuple) -> None:
    return img.resize(new_size, Image.BILINEAR)

def resize_image_kornia(img: np.ndarray, new_size: tuple) -> None:
    return kornia_rs.resize(img, new_size, "bilinear")

tests = [
    {
        "name": "OpenCV",
        "stmt": "resize_image_opencv(img, new_size)",
        "setup": "from __main__ import resize_image_opencv, img, new_size",
        "globals": {"img": img, "new_size": new_size},
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
        "setup": "from __main__ import resize_image_kornia, img, new_size",
        "globals": {"img": img, "new_size": new_size},
    },
]

for test in tests:
    timer = timeit.Timer(
        stmt=test["stmt"], setup=test["setup"], globals=test["globals"]
    )
    print(f"{test['name']}: {timer.timeit(N)/ N * 1e3:.2f} ms")
