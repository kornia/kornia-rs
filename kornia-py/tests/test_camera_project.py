import numpy as np

import kornia_rs as K


def _assert_jacobian_matches_finite_differences(cam, point: np.ndarray, atol_scale: float = 1e-5):
    points = np.array([point], dtype=np.float64)
    pixels, jacobians = cam.project_with_jacobian(points)
    analytic = jacobians[0]
    h = 1e-6
    for col in range(3):
        delta = np.zeros(3, dtype=np.float64)
        delta[col] = h
        plus = cam.project(np.array([point + delta], dtype=np.float64))[0]
        minus = cam.project(np.array([point - delta], dtype=np.float64))[0]
        numeric = (plus - minus) / (2.0 * h)
        tolerance = atol_scale * max(np.linalg.norm(analytic[:, col]), 1.0)
        np.testing.assert_allclose(analytic[:, col], numeric, atol=tolerance, rtol=0.0)


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


def test_pinhole_project_with_jacobian_shape_and_dtype():
    cam = K.k3d.PinholeCamera(fx=400.0, fy=410.0, cx=320.0, cy=240.0)
    points = np.array(
        [
            [0.0, 0.0, 1.0],
            [0.2, -0.1, 2.0],
        ],
        dtype=np.float64,
    )

    pixels, jacobians = cam.project_with_jacobian(points)

    assert pixels.shape == (2, 2)
    assert jacobians.shape == (2, 2, 3)
    assert pixels.dtype == np.float64
    assert jacobians.dtype == np.float64


def test_pinhole_project_with_jacobian_invalid_point_maps_to_nan():
    cam = K.k3d.PinholeCamera(fx=500.0, fy=500.0, cx=320.0, cy=240.0)
    points = np.array(
        [
            [0.0, 0.0, 1.0],
            [1.0, 0.0, -1.0],
        ],
        dtype=np.float64,
    )

    pixels, jacobians = cam.project_with_jacobian(points)

    assert np.isfinite(pixels[0]).all()
    assert np.isfinite(jacobians[0]).all()
    assert np.isnan(pixels[1]).all()
    assert np.isnan(jacobians[1]).all()


def test_pinhole_project_with_jacobian_matches_finite_differences():
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
    for point in (
        np.array([0.0, 0.0, 2.0], dtype=np.float64),
        np.array([0.4, -0.3, 1.5], dtype=np.float64),
        np.array([-0.6, 0.35, 1.0], dtype=np.float64),
    ):
        _assert_jacobian_matches_finite_differences(cam, point)


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


def test_pinhole_undistort_shape_and_dtype():
    cam = K.k3d.PinholeCamera(fx=400.0, fy=410.0, cx=320.0, cy=240.0)
    pixels = np.array(
        [
            [320.0, 240.0],
            [400.0, 200.0],
        ],
        dtype=np.float64,
    )

    undistorted = cam.undistort(pixels)

    assert undistorted.shape == (2, 2)
    assert undistorted.dtype == np.float64


def test_pinhole_undistort_zero_distortion_is_identity():
    cam = K.k3d.PinholeCamera(fx=500.0, fy=500.0, cx=320.0, cy=240.0)
    pixels = np.array(
        [
            [320.0, 240.0],
            [200.0, 300.0],
            [500.0, 120.0],
        ],
        dtype=np.float64,
    )

    undistorted = cam.undistort(pixels)

    np.testing.assert_allclose(undistorted, pixels, atol=1e-12)


def test_pinhole_undistort_rejects_wrong_column_count():
    cam = K.k3d.PinholeCamera(fx=500.0, fy=500.0, cx=320.0, cy=240.0)
    pixels = np.array([[0.0, 1.0, 2.0]], dtype=np.float64)

    with np.testing.assert_raises(ValueError):
        cam.undistort(pixels)


def test_pinhole_undistort_invalid_pixel_maps_to_nan():
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

    undistorted = cam.undistort(pixels)

    assert np.isfinite(undistorted[0]).all()
    assert np.isnan(undistorted[1]).all()


def test_pinhole_undistort_project_roundtrip_corner_pixels():
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

    ideal = cam.undistort(corners)
    rays = np.column_stack(
        [
            (ideal[:, 0] - 367.215) / 458.654,
            (ideal[:, 1] - 248.375) / 457.296,
            np.ones(ideal.shape[0], dtype=np.float64),
        ]
    )
    pixels_back = cam.project(rays)

    np.testing.assert_allclose(pixels_back, corners, atol=1e-6)


def test_fisheye_project_shape_and_dtype():
    cam = K.k3d.FisheyeCamera(fx=461.64, fy=459.72, cx=732.95, cy=720.54)
    points = np.array(
        [
            [0.1, -0.2, 1.0],
            [0.5, 0.3, -0.2],
        ],
        dtype=np.float64,
    )

    pixels = cam.project(points)

    assert pixels.shape == (2, 2)
    assert pixels.dtype == np.float64


def test_fisheye_project_with_jacobian_shape_and_dtype():
    cam = K.k3d.FisheyeCamera(fx=461.64, fy=459.72, cx=732.95, cy=720.54)
    points = np.array(
        [
            [0.1, -0.2, 1.0],
            [0.5, 0.3, -0.2],
        ],
        dtype=np.float64,
    )

    pixels, jacobians = cam.project_with_jacobian(points)

    assert pixels.shape == (2, 2)
    assert jacobians.shape == (2, 2, 3)
    assert pixels.dtype == np.float64
    assert jacobians.dtype == np.float64


def test_fisheye_project_with_jacobian_invalid_point_maps_to_nan():
    cam = K.k3d.FisheyeCamera(fx=461.64, fy=459.72, cx=732.95, cy=720.54)
    points = np.array(
        [
            [0.1, -0.2, 1.0],
            [0.0, 0.0, -1.0],
        ],
        dtype=np.float64,
    )

    pixels, jacobians = cam.project_with_jacobian(points)

    assert np.isfinite(pixels[0]).all()
    assert np.isfinite(jacobians[0]).all()
    assert np.isnan(pixels[1]).all()
    assert np.isnan(jacobians[1]).all()


def test_fisheye_project_with_jacobian_matches_finite_differences():
    cam = K.k3d.FisheyeCamera(
        fx=461.64,
        fy=459.72,
        cx=732.95,
        cy=720.54,
        k1=0.0344,
        k2=-0.0216,
        k3=0.0031,
        k4=-0.0005,
    )
    for point in (
        np.array([0.0, 0.0, 2.0], dtype=np.float64),
        np.array([0.4, -0.3, 1.5], dtype=np.float64),
        np.array([3.0, 1.0, 0.5], dtype=np.float64),
        np.array([1.0, 0.5, -0.2], dtype=np.float64),
    ):
        _assert_jacobian_matches_finite_differences(cam, point)


def test_fisheye_project_optical_center_behind_maps_to_nan():
    cam = K.k3d.FisheyeCamera(fx=461.64, fy=459.72, cx=732.95, cy=720.54)
    points = np.array(
        [
            [0.0, 0.0, 1.0],
            [0.0, 0.0, -1.0],
        ],
        dtype=np.float64,
    )

    pixels = cam.project(points)

    assert np.isfinite(pixels[0]).all()
    assert np.isnan(pixels[1]).all()


def test_fisheye_unproject_shape_and_dtype():
    cam = K.k3d.FisheyeCamera(fx=461.64, fy=459.72, cx=732.95, cy=720.54)
    pixels = np.array(
        [
            [732.95, 720.54],
            [900.0, 650.0],
        ],
        dtype=np.float64,
    )

    rays = cam.unproject(pixels)

    assert rays.shape == (2, 3)
    assert rays.dtype == np.float64


def test_fisheye_unproject_returns_unit_bearings():
    cam = K.k3d.FisheyeCamera(fx=461.64, fy=459.72, cx=732.95, cy=720.54)
    pixels = np.array(
        [
            [732.95, 720.54],
            [900.0, 650.0],
            [650.0, 810.0],
        ],
        dtype=np.float64,
    )

    rays = cam.unproject(pixels)
    norms = np.linalg.norm(rays, axis=1)

    np.testing.assert_allclose(norms, np.ones_like(norms), atol=1e-12)


def test_fisheye_unproject_rejects_wrong_column_count():
    cam = K.k3d.FisheyeCamera(fx=461.64, fy=459.72, cx=732.95, cy=720.54)
    pixels = np.array([[0.0, 1.0, 2.0]], dtype=np.float64)

    with np.testing.assert_raises(ValueError):
        cam.unproject(pixels)


def test_fisheye_unproject_invalid_pixel_maps_to_nan():
    cam = K.k3d.FisheyeCamera(
        fx=461.64,
        fy=459.72,
        cx=732.95,
        cy=720.54,
        k1=0.0344,
        k2=-0.0216,
        k3=0.0031,
        k4=-0.0005,
    )
    pixels = np.array(
        [
            [732.95, 720.54],
            [732.95 + 10.0 * 461.64, 720.54],
        ],
        dtype=np.float64,
    )

    rays = cam.unproject(pixels)

    assert np.isfinite(rays[0]).all()
    assert np.isnan(rays[1]).all()


def test_fisheye_project_unproject_roundtrip_bearing():
    cam = K.k3d.FisheyeCamera(
        fx=461.64,
        fy=459.72,
        cx=732.95,
        cy=720.54,
        k1=0.0344,
        k2=-0.0216,
        k3=0.0031,
        k4=-0.0005,
    )
    points = np.array(
        [
            [0.4, -0.3, 1.5],
            [3.0, 1.0, 0.5],
            [1.0, 0.5, -0.2],
        ],
        dtype=np.float64,
    )

    pixels = cam.project(points)
    rays = cam.unproject(pixels)
    target = points / np.linalg.norm(points, axis=1, keepdims=True)

    np.testing.assert_allclose(rays, target, atol=1e-9)


def test_runtime_camera_pinhole_dispatch_matches_pinhole_camera():
    pinhole = K.k3d.PinholeCamera(
        fx=458.654,
        fy=457.296,
        cx=367.215,
        cy=248.375,
        k1=-0.28340811,
        k2=0.07395907,
        p1=0.00019359,
        p2=1.76187114e-05,
    )
    camera = K.k3d.Camera.pinhole(
        fx=458.654,
        fy=457.296,
        cx=367.215,
        cy=248.375,
        k1=-0.28340811,
        k2=0.07395907,
        p1=0.00019359,
        p2=1.76187114e-05,
    )

    points = np.array(
        [
            [0.4, -0.3, 1.5],
            [1.0, 0.5, -0.2],
        ],
        dtype=np.float64,
    )
    pixels = np.array(
        [
            [0.0, 0.0],
            [751.0, 479.0],
        ],
        dtype=np.float64,
    )

    np.testing.assert_allclose(camera.project(points), pinhole.project(points), equal_nan=True)
    runtime_pixels, runtime_jacobians = camera.project_with_jacobian(points)
    pinhole_pixels, pinhole_jacobians = pinhole.project_with_jacobian(points)
    np.testing.assert_allclose(runtime_pixels, pinhole_pixels, equal_nan=True)
    np.testing.assert_allclose(runtime_jacobians, pinhole_jacobians, equal_nan=True)
    np.testing.assert_allclose(camera.unproject(pixels), pinhole.unproject(pixels), equal_nan=True)


def test_runtime_camera_fisheye_dispatch_matches_fisheye_camera():
    fisheye = K.k3d.FisheyeCamera(
        fx=461.64,
        fy=459.72,
        cx=732.95,
        cy=720.54,
        k1=0.0344,
        k2=-0.0216,
        k3=0.0031,
        k4=-0.0005,
    )
    camera = K.k3d.Camera.fisheye(
        fx=461.64,
        fy=459.72,
        cx=732.95,
        cy=720.54,
        k1=0.0344,
        k2=-0.0216,
        k3=0.0031,
        k4=-0.0005,
    )

    points = np.array(
        [
            [0.4, -0.3, 1.5],
            [1.0, 0.5, -0.2],
        ],
        dtype=np.float64,
    )
    pixels = np.array(
        [
            [732.95, 720.54],
            [732.95 + 10.0 * 461.64, 720.54],
        ],
        dtype=np.float64,
    )

    np.testing.assert_allclose(camera.project(points), fisheye.project(points), equal_nan=True)
    runtime_pixels, runtime_jacobians = camera.project_with_jacobian(points)
    fisheye_pixels, fisheye_jacobians = fisheye.project_with_jacobian(points)
    np.testing.assert_allclose(runtime_pixels, fisheye_pixels, equal_nan=True)
    np.testing.assert_allclose(runtime_jacobians, fisheye_jacobians, equal_nan=True)
    np.testing.assert_allclose(camera.unproject(pixels), fisheye.unproject(pixels), equal_nan=True)
