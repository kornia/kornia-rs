use kornia_imgproc::undistort::undistort_points;
use ndarray::{array, Array1, Array2};

#[test]
fn test_undistort_points_identity() {
    // Test case with zero distortion and identity matrices
    let src_points = array![[100.0, 200.0], [300.0, 400.0]];
    let camera_matrix = array![
        [500.0, 0.0, 320.0],
        [0.0, 500.0, 240.0],
        [0.0, 0.0, 1.0]
    ];
    let dist_coeffs = Array1::<f64>::zeros(5); // No distortion
    let r_matrix = None; // No rectification

    // With no P matrix, result should be normalized coordinates
    let p_matrix = None;
    let dst_points =
        undistort_points(&src_points, &camera_matrix, &dist_coeffs, &r_matrix, &p_matrix).unwrap();

    // Expected normalized coordinates: (u-cx)/fx, (v-cy)/fy
    let expected = array![
        [(100.0 - 320.0) / 500.0, (200.0 - 240.0) / 500.0],
        [(300.0 - 320.0) / 500.0, (400.0 - 240.0) / 500.0]
    ];

    assert!(
        dst_points
            .iter()
            .zip(expected.iter())
            .all(|(a, b)| (a - b).abs() < 1e-6),
        "Expected {:?}, but got {:?}",
        expected,
        dst_points
    );
}

#[test]
fn test_undistort_points_with_p() {
    // If P is the same as K, and no distortion/rectification, output should equal input
    let src_points = array![[100.0, 200.0]];
    let camera_matrix = array![
        [500.0, 0.0, 320.0],
        [0.0, 500.0, 240.0],
        [0.0, 0.0, 1.0]
    ];
    let dist_coeffs = Array1::<f64>::zeros(5);
    let r_matrix = None;
    let p_matrix = Some(camera_matrix.clone());

    let dst_points =
        undistort_points(&src_points, &camera_matrix, &dist_coeffs, &r_matrix, &p_matrix).unwrap();

    assert!(
        dst_points
            .iter()
            .zip(src_points.iter())
            .all(|(a, b)| (a - b).abs() < 1e-6),
        "Expected {:?}, but got {:?}",
        src_points,
        dst_points
    );
}

#[test]
fn test_undistort_with_distortion() {
    // Test with known distortion values to verify the algorithm
    // This requires calculating a forward distortion to get a test point.
    let camera_matrix = array![
        [1000.0, 0.0, 500.0],
        [0.0, 1000.0, 500.0],
        [0.0, 0.0, 1.0]
    ];
    // k1 = 0.5, k2 = -0.2, p1 = 0.01, p2 = 0.02
    let dist_coeffs = array![0.5, -0.2, 0.01, 0.02, 0.0];

    // Let's take an ideal point in normalized coordinates
    // FIX: Explicitly define the float type as f64 to resolve ambiguity.
    let ideal_point_normalized = array![[0.1_f64, 0.2_f64]];
    let (x, y) = (ideal_point_normalized[[0, 0]], ideal_point_normalized[[0, 1]]);

    // Apply the forward distortion model manually
    let r2 = x.powi(2) + y.powi(2);
    let r4 = r2 * r2;
    let radial_dist = 1.0 + dist_coeffs[0] * r2 + dist_coeffs[1] * r4;
    let x_distorted_norm = x * radial_dist + (2.0 * dist_coeffs[2] * x * y + dist_coeffs[3] * (r2 + 2.0 * x * x));
    let y_distorted_norm = y * radial_dist + (dist_coeffs[2] * (r2 + 2.0 * y * y) + 2.0 * dist_coeffs[3] * x * y);

    // Convert distorted normalized coordinates to pixel coordinates
    let (fx, fy, cx, cy) = (
        camera_matrix[[0, 0]],
        camera_matrix[[1, 1]],
        camera_matrix[[0, 2]],
        camera_matrix[[1, 2]],
    );
    let u_distorted = x_distorted_norm * fx + cx;
    let v_distorted = y_distorted_norm * fy + cy;
    let src_points = array![[u_distorted, v_distorted]];

    let dst_points_normalized = undistort_points(
        &src_points,
        &camera_matrix,
        &dist_coeffs,
        &None,
        &None, 
    )
    .unwrap();

    assert!(
        dst_points_normalized
            .iter()
            .zip(ideal_point_normalized.iter())
            .all(|(a, b)| (a - b).abs() < 1e-6),
        "Distortion test failed. Expected normalized {:?}, got {:?}",
        ideal_point_normalized,
        dst_points_normalized
    );
}

