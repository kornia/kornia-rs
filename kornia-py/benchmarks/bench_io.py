import timeit
from pathlib import Path

import cv2
from PIL import Image as PILImage
import numpy as np

from kornia_rs.image import Image

# tensorflow is heavy and rarely installed alongside cv2/PIL/kornia. Guard
# the import so the bench still runs the OpenCV/Pillow/Kornia rows without
# it; the TF row is appended only if `tf.keras.utils.load_img` is available.
try:
    import tensorflow as tf
    HAVE_TF = True
except ImportError:
    HAVE_TF = False

image_path = str(Path(__file__).resolve().parents[2] / "tests" / "data" / "dog.jpeg")
N = 5000  # number of iterations


def read_image_opencv(image_path: str) -> None:
    return cv2.imread(image_path)


def read_image_pillow(image_path: str) -> None:
    return np.array(PILImage.open(image_path))


def read_image_kornia(image_path: str) -> None:
    return Image.load(image_path)


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
]
if HAVE_TF:
    tests.append({
        "name": "Tensorflow",
        "stmt": "read_image_tensorflow(image_path)",
        "setup": "from __main__ import read_image_tensorflow",
        "globals": {"image_path": image_path},
    })

for test in tests:
    timer = timeit.Timer(
        stmt=test["stmt"], setup=test["setup"], globals=test["globals"]
    )
    print(f"{test['name']}: {timer.timeit(N)/ N * 1e3:.2f} ms")
