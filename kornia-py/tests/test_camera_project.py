import numpy as np

import kornia_rs as K


def test_pinhole_project_shape_and_dtype():
    cam = K.k3d.PinholeCamera(fx=400.0, fy=410.0, cx=320.0, cy=240.0)
    points = np.array(
        [
            [0.0, 0.0, 1.0],
            [0.2, -0.1, 2.0],
        ],
        dtype=np.float64,
    )

    pixels = cam.project(points)

    assert pixels.shape == (2, 2)
    assert pixels.dtype == np.float64


def test_pinhole_project_zero_distortion_center_projection():
    cam = K.k3d.PinholeCamera(fx=500.0, fy=500.0, cx=320.0, cy=240.0)
    points = np.array([[0.0, 0.0, 1.0]], dtype=np.float64)

    pixels = cam.project(points)

    np.testing.assert_allclose(pixels[0], np.array([320.0, 240.0], dtype=np.float64), atol=1e-12)


def test_pinhole_project_invalid_depth_maps_to_nan():
    cam = K.k3d.PinholeCamera(fx=500.0, fy=500.0, cx=320.0, cy=240.0)
    points = np.array(
        [
            [0.0, 0.0, 1.0],
            [1.0, 0.0, -1.0],
        ],
        dtype=np.float64,
    )

    pixels = cam.project(points)

    assert np.isfinite(pixels[0]).all()
    assert np.isnan(pixels[1]).all()


def test_pinhole_project_rejects_wrong_column_count():
    cam = K.k3d.PinholeCamera(fx=500.0, fy=500.0, cx=320.0, cy=240.0)
    points = np.array([[0.0, 1.0]], dtype=np.float64)

    with np.testing.assert_raises(ValueError):
        cam.project(points)
