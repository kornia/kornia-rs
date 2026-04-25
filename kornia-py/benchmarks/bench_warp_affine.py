import timeit
from pathlib import Path

import cv2
from PIL import Image as PILImage
import kornia_rs
import numpy as np

from kornia_rs.image import Image

image_path = str(Path(__file__).resolve().parents[2] / "tests" / "data" / "dog.jpeg")
img_kornia = Image.load(image_path)
img = img_kornia.to_numpy()
img_pil = PILImage.open(image_path)
height, width, _ = img.shape
M = cv2.getRotationMatrix2D((width / 2, height / 2), 45.0, 1.0)
M_tuple = tuple(M.flatten())
N = 5000  # number of iterations


def warp_affine_opencv(img: np.ndarray, M: np.ndarray) -> None:
    return cv2.warpAffine(img, M, (img.shape[1], img.shape[0]), flags=cv2.INTER_LINEAR)

def warp_affine_pil(img: PILImage.Image, M_tuple: tuple) -> None:
    return img.transform(img.size, PILImage.Transform.AFFINE, M_tuple, PILImage.BILINEAR)

def warp_affine_kornia(img: np.ndarray, M_tuple: tuple) -> None:
    return kornia_rs.warp_affine(img, M_tuple, img.shape[:2], "bilinear")

tests = [
    {
        "name": "OpenCV",
        "stmt": "warp_affine_opencv(img, M)",
        "setup": "from __main__ import warp_affine_opencv, img, M",
        "globals": {"img": img, "M": M},
    },
    {
        "name": "PIL",
        "stmt": "warp_affine_pil(img_pil, M_tuple)",
        "setup": "from __main__ import warp_affine_pil, img_pil, M_tuple",
        "globals": {"img_pil": img_pil, "M_tuple": M_tuple},
    },
    {
        "name": "Kornia",
        "stmt": "warp_affine_kornia(img, M_tuple)",
        "setup": "from __main__ import warp_affine_kornia, img, M_tuple",
        "globals": {"img": img, "M_tuple": M_tuple},
    },
]

for test in tests:
    timer = timeit.Timer(
        stmt=test["stmt"], setup=test["setup"], globals=test["globals"]
    )
    print(f"{test['name']}: {timer.timeit(N)/ N * 1e3:.2f} ms")
