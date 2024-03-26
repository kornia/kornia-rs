import timeit

import cv2
from PIL import Image
import kornia_rs
import numpy as np
import tensorflow as tf

image_path = "tests/data/dog.jpeg"
N = 5000  # number of iterations


def read_image_opencv(image_path: str) -> None:
    return cv2.imread(image_path)


def read_image_pillow(image_path: str) -> None:
    return np.array(Image.open(image_path))


def read_image_kornia(image_path: str) -> None:
    return kornia_rs.read_image_jpeg(image_path)

def read_image_tensorflow(image_path: str) -> None:
    return tf.keras.utils.load_img(image_path)


tests = [
    {
        "name": "OpenCV",
        "stmt": "read_image_opencv(image_path)",
        "setup": "from __main__ import read_image_opencv",
        "globals": {"image_path": image_path},
    },
    {
        "name": "Pillow",
        "stmt": "read_image_pillow(image_path)",
        "setup": "from __main__ import read_image_pillow",
        "globals": {"image_path": image_path},
    },
    {
        "name": "Kornia",
        "stmt": "read_image_kornia(image_path)",
        "setup": "from __main__ import read_image_kornia",
        "globals": {"image_path": image_path},
    },
    {
        "name": "Tensorflow",
        "stmt": "read_image_tensorflow(image_path)",
        "setup": "from __main__ import read_image_tensorflow",
        "globals": {"image_path": image_path},
    },
]

for test in tests:
    timer = timeit.Timer(
        stmt=test["stmt"], setup=test["setup"], globals=test["globals"]
    )
    print(f"{test['name']}: {timer.timeit(N)/ N * 1e3:.2f} ms")
