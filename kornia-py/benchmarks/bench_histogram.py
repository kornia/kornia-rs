import timeit
from pathlib import Path

import cv2
import kornia_rs as K
import numpy as np

from kornia_rs.image import Image

image_path = str(Path(__file__).resolve().parents[2] / "tests" / "data" / "dog.jpeg")
N = 5000  # number of iterations

img = np.ascontiguousarray(Image.load(image_path).to_numpy()[..., :1])


# 0.04 ms :)
def hist_opencv(image: np.ndarray) -> np.ndarray:
    return cv2.calcHist([image], [0], None, [256], [0, 256])


# 0.17 ms :(
def hist_kornia(image: np.ndarray) -> np.ndarray:
    return K.compute_histogram(image, nbins=256)


tests = [
    {
        "name": "OpenCV",
        "stmt": "hist_opencv(image)",
        "setup": "from __main__ import hist_opencv",
        "globals": {"image": img},
    },
    {
        "name": "Kornia",
        "stmt": "hist_kornia(image)",
        "setup": "from __main__ import hist_kornia",
        "globals": {"image": img},
    },
]

for test in tests:
    timer = timeit.Timer(
        stmt=test["stmt"], setup=test["setup"], globals=test["globals"]
    )
    print(f"{test['name']}: {timer.timeit(N)/ N * 1e3:.2f} ms")
