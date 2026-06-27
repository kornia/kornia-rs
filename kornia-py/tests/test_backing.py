import numpy as np
from kornia_rs.image import Image


def test_from_numpy_zero_copy_shares_memory():
    arr = np.zeros((8, 8, 3), dtype=np.uint8)
    img = Image(arr)                      # borrow by default
    arr[1, 2, 0] = 200
    assert img.numpy()[1, 2, 0] == 200    # shares memory
    img.numpy()[3, 3, 1] = 50
    assert arr[3, 3, 1] == 50


def test_from_numpy_copy_is_independent():
    arr = np.zeros((4, 4, 3), dtype=np.uint8)
    img = Image.from_numpy(arr, copy=True)
    arr[0, 0, 0] = 99
    assert img.numpy()[0, 0, 0] == 0


def test_owned_output_survives_source_free():
    arr = np.random.randint(0, 255, (16, 16, 3), np.uint8)
    img = Image(arr)
    out = img.resize(8, 8)                # owned output
    del arr, img
    assert out.numpy().shape == (8, 8, 3)  # still valid
