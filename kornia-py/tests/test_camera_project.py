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


def test_pinhole_unproject_shape_and_dtype():
    cam = K.k3d.PinholeCamera(fx=400.0, fy=410.0, cx=320.0, cy=240.0)
    pixels = np.array(
        [
            [320.0, 240.0],
            [410.0, 200.0],
        ],
        dtype=np.float64,
    )

    rays = cam.unproject(pixels)

    assert rays.shape == (2, 3)
    assert rays.dtype == np.float64


def test_pinhole_unproject_center_is_optical_axis():
    cam = K.k3d.PinholeCamera(fx=500.0, fy=500.0, cx=320.0, cy=240.0)
    pixels = np.array([[320.0, 240.0]], dtype=np.float64)

    rays = cam.unproject(pixels)

    np.testing.assert_allclose(rays[0], np.array([0.0, 0.0, 1.0], dtype=np.float64), atol=1e-12)


def test_pinhole_unproject_returns_unit_bearings():
    cam = K.k3d.PinholeCamera(fx=500.0, fy=500.0, cx=320.0, cy=240.0)
    pixels = np.array(
        [
            [320.0, 240.0],
            [200.0, 300.0],
            [500.0, 120.0],
        ],
        dtype=np.float64,
    )

    rays = cam.unproject(pixels)
    norms = np.linalg.norm(rays, axis=1)

    np.testing.assert_allclose(norms, np.ones_like(norms), atol=1e-12)


def test_pinhole_unproject_rejects_wrong_column_count():
    cam = K.k3d.PinholeCamera(fx=500.0, fy=500.0, cx=320.0, cy=240.0)
    pixels = np.array([[0.0, 1.0, 2.0]], dtype=np.float64)

    with np.testing.assert_raises(ValueError):
        cam.unproject(pixels)


def test_pinhole_unproject_invalid_pixel_maps_to_nan():
    cam = K.k3d.PinholeCamera(
        fx=500.0,
        fy=500.0,
        cx=376.0,
        cy=240.0,
        k1=-0.28,
        k2=0.0,
        p1=0.0,
        p2=0.0,
    )
    pixels = np.array(
        [
            [400.0, 250.0],
            [0.0, 0.0],
        ],
        dtype=np.float64,
    )

    rays = cam.unproject(pixels)

    assert np.isfinite(rays[0]).all()
    assert np.isnan(rays[1]).all()


def test_pinhole_project_unproject_corner_roundtrip():
    cam = K.k3d.PinholeCamera(
        fx=458.654,
        fy=457.296,
        cx=367.215,
        cy=248.375,
        k1=-0.28340811,
        k2=0.07395907,
        p1=0.00019359,
        p2=1.76187114e-05,
    )
    corners = np.array(
        [
            [0.0, 0.0],
            [751.0, 0.0],
            [0.0, 479.0],
            [751.0, 479.0],
        ],
        dtype=np.float64,
    )

    rays = cam.unproject(corners)
    pixels_back = cam.project(rays)

    np.testing.assert_allclose(pixels_back, corners, atol=1e-6)
