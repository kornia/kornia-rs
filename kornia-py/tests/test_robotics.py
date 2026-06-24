"""Robotics zero-copy path tests — ROS2-style bytearray ingest via from_buffer.

Task #23: ROS2 bgr8 -> zero-copy Image -> owned output.
"""

import numpy as np
import pytest
from kornia_rs.image import Image, ColorSpace


def test_from_buffer_zero_copy_bgr8():
    """from_buffer borrows the bytearray without copying; mutations are visible."""
    h, w = 64, 96
    src = np.random.randint(0, 255, (h, w, 3), np.uint8)
    payload = bytearray(src.tobytes())  # transient ROS2-style buffer
    img = Image.from_buffer(
        payload, width=w, height=h, channels=3, dtype="uint8", color_space=ColorSpace.Bgr
    )
    assert img.color_space == ColorSpace.Bgr
    # zero-copy: mutate the payload, verify change is visible in the image
    original_val = src[0, 0, 0]
    mutated_val = original_val ^ 0xFF
    payload[0] = mutated_val
    assert np.asarray(img)[0, 0, 0] == mutated_val


def test_from_buffer_to_owned_tensor_survives_source_free():
    """Owning output produced from a borrowed image is independent of the source."""
    h, w = 64, 96
    payload = bytearray(np.random.randint(0, 255, (h, w, 3), np.uint8).tobytes())
    img = Image.from_buffer(
        payload, width=w, height=h, channels=3, color_space=ColorSpace.Bgr
    )
    # cvt_color produces an OWNED output — independent of payload
    rgb = img.cvt_color(ColorSpace.Rgb)
    # Free the source; rgb must remain valid
    del payload, img
    arr = rgb.numpy()
    assert arr.shape == (h, w, 3)
    assert rgb.data_ptr != 0
