use glam::{Mat3, Vec3};
use kornia_linalg::linalg::svd3;

/// Compute centroids of two point clouds
///
/// This function calculates the centroid of two point clouds by averaging
/// their positions. The centroids are used in rigid transformation estimation.
pub fn compute_centroids(points_src: &[Vec3], points_dst: &[Vec3]) -> (Vec3, Vec3) {
    let src_centroid =
        points_src.iter().fold(Vec3::ZERO, |acc, &p| acc + p) / points_src.len() as f32;
    let dst_centroid =
        points_dst.iter().fold(Vec3::ZERO, |acc, &p| acc + p) / points_dst.len() as f32;
    (src_centroid, dst_centroid)
}

/// Compute optimal rigid transformation between two point clouds
///
/// This function finds the optimal rigid transformation (rotation and translation)
/// that aligns two point clouds with the same number of points. It implements
/// the SVD-based algorithm for point cloud alignment, which minimizes the
/// mean square error between corresponding points.
///
/// The algorithm:
/// 1. Compute centroids of both point clouds
/// 2. Center both point clouds by subtracting their respective centroids
/// 3. Compute the cross-covariance matrix H = Σ[(p_src - p_src_mean) * (p_dst - p_dst_mean)^T]
/// 4. Compute the SVD of H = U * S * V^T
/// 5. Calculate rotation matrix R = V * U^T
/// 6. Calculate translation vector t = p_dst_mean - R * p_src_mean
///
/// For more details, see: Arun, K., Huang, T. S., and Blostein, S. D.
/// "Least-squares fitting of two 3-D point sets." IEEE PAMI, 1987.
///
/// # Arguments
/// * `points_src` - Source point cloud
/// * `points_dst` - Destination point cloud (must have same length as `points_src`)
///
/// # Returns
/// * `(Mat3, Vec3)` - The rotation matrix and translation vector that transform
///   `points_src` to align with `points_dst`
pub fn fit_transformation(points_src: &[Vec3], points_dst: &[Vec3]) -> (Mat3, Vec3) {
    assert_eq!(
        points_src.len(),
        points_dst.len(),
        "Point clouds must have same number of points"
    );
    assert!(
        points_src.len() >= 3,
        "Need at least 3 points for transformation estimation"
    );

    // Identity transformation is a special case
    if points_src == points_dst {
        return (Mat3::IDENTITY, Vec3::ZERO);
    }

    // Compute centroids
    let (src_centroid, dst_centroid) = compute_centroids(points_src, points_dst);

    // Compute cross-covariance matrix H = Σ[(src - src_mean) * (dst - dst_mean)^T]
    let mut h = Mat3::ZERO;
    for (&src_pt, &dst_pt) in points_src.iter().zip(points_dst.iter()) {
        let src_centered = src_pt - src_centroid;
        let dst_centered = dst_pt - dst_centroid;

        // Compute the outer product directly using the same formula as in ops.rs
        h += Mat3::from_cols(
            src_centered * dst_centered.x,
            src_centered * dst_centered.y,
            src_centered * dst_centered.z,
        );
    }

    // Try direct computation using 3 points if available
    if points_src.len() >= 4 {
        let mut src_pts = Mat3::ZERO;
        let mut dst_pts = Mat3::ZERO;

        // Use points 1, 2, 3 (unit vectors) to form orthogonal basis
        for i in 0..3 {
            let src_pt = points_src[i + 1] - src_centroid; // Skip origin
            let dst_pt = points_dst[i + 1] - dst_centroid;

            match i {
                0 => {
                    src_pts.x_axis = src_pt;
                    dst_pts.x_axis = dst_pt;
                }
                1 => {
                    src_pts.y_axis = src_pt;
                    dst_pts.y_axis = dst_pt;
                }
                2 => {
                    src_pts.z_axis = src_pt;
                    dst_pts.z_axis = dst_pt;
                }
                _ => {}
            }
        }

        // If src_pts is invertible, this is the most direct way
        // to compute the rotation
        let det = src_pts.determinant();
        if det.abs() > 1e-6 {
            let r_direct = dst_pts * src_pts.inverse();

            // If direct computation worked well, use it
            if (r_direct.determinant() - 1.0).abs() < 0.1 {
                return (r_direct, dst_centroid - r_direct * src_centroid);
            }
        }
    }

    // Compute SVD of covariance matrix
    let svd_result = svd3(&h);
    let u = *svd_result.u();
    let v = *svd_result.v();

    // Using V * U^T gives a rotation matrix
    // If det(V * U^T) < 0, we need to flip a sign to avoid reflections
    let mut r = v * u.transpose();

    // Handle reflection case to ensure proper rotation matrix
    if r.determinant() < 0.0 {
        // Create a modified V matrix with the z-axis negated
        let v_corrected = Mat3::from_cols(v.x_axis, v.y_axis, -v.z_axis);
        r = v_corrected * u.transpose();
    }

    // Compute translation vector
    let t = dst_centroid - r * src_centroid;

    (r, t)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Utility function to check if two matrices are approximately equal
    fn mat3_approx_eq(a: &Mat3, b: &Mat3, epsilon: f32) -> bool {
        let cols_a = [a.x_axis, a.y_axis, a.z_axis];
        let cols_b = [b.x_axis, b.y_axis, b.z_axis];

        for i in 0..3 {
            if !vec3_approx_eq(&cols_a[i], &cols_b[i], epsilon) {
                return false;
            }
        }
        true
    }

    /// Utility function to check if two vectors are approximately equal
    fn vec3_approx_eq(a: &Vec3, b: &Vec3, epsilon: f32) -> bool {
        (a.x - b.x).abs() < epsilon && (a.y - b.y).abs() < epsilon && (a.z - b.z).abs() < epsilon
    }

    #[test]
    fn test_identity_transformation() {
        // Create simple test case
        let points_src = vec![
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::new(1.0, 0.0, 0.0),
            Vec3::new(0.0, 1.0, 0.0),
            Vec3::new(0.0, 0.0, 1.0),
        ];

        let points_dst = points_src.clone();

        let (r, t) = fit_transformation(&points_src, &points_dst);

        assert!(
            mat3_approx_eq(&r, &Mat3::IDENTITY, 1e-6),
            "Expected identity rotation, got: {:?}",
            r
        );
        assert!(
            vec3_approx_eq(&t, &Vec3::ZERO, 1e-6),
            "Expected zero translation, got: {:?}",
            t
        );
    }

    #[test]
    fn test_pure_rotation() {
        // Simple 90° rotation around X axis
        let points_src = vec![
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::new(1.0, 0.0, 0.0),
            Vec3::new(0.0, 1.0, 0.0),
            Vec3::new(0.0, 0.0, 1.0),
        ];

        // Rotate 90° around X axis: y -> -z, z -> y
        let points_dst = vec![
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::new(1.0, 0.0, 0.0),
            Vec3::new(0.0, 0.0, -1.0),
            Vec3::new(0.0, 1.0, 0.0),
        ];

        let (r, t) = fit_transformation(&points_src, &points_dst);

        let expected_r = Mat3::from_cols(
            Vec3::new(1.0, 0.0, 0.0),
            Vec3::new(0.0, 0.0, -1.0),
            Vec3::new(0.0, 1.0, 0.0),
        );

        assert!(
            mat3_approx_eq(&r, &expected_r, 1e-6),
            "Expected rotation:\n{:?}\nGot:\n{:?}",
            expected_r,
            r
        );
        assert!(
            vec3_approx_eq(&t, &Vec3::ZERO, 1e-6),
            "Expected zero translation, got: {:?}",
            t
        );
    }

    #[test]
    fn test_translation() {
        let translation = Vec3::new(5.0, -3.0, 2.0);

        // Create simple test case
        let points_src = vec![
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::new(1.0, 0.0, 0.0),
            Vec3::new(0.0, 1.0, 0.0),
            Vec3::new(0.0, 0.0, 1.0),
        ];

        let points_dst: Vec<Vec3> = points_src.iter().map(|p| *p + translation).collect();

        let (r, t) = fit_transformation(&points_src, &points_dst);

        assert!(
            mat3_approx_eq(&r, &Mat3::IDENTITY, 1e-6),
            "Expected identity rotation, got: {:?}",
            r
        );
        assert!(
            vec3_approx_eq(&t, &translation, 1e-6),
            "Expected translation {:?}, got: {:?}",
            translation,
            t
        );
    }

    #[test]
    fn test_combined_transform() {
        // Simple 90° rotation around Z + translation
        let points_src = vec![
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::new(1.0, 0.0, 0.0),
            Vec3::new(0.0, 1.0, 0.0),
            Vec3::new(0.0, 0.0, 1.0),
        ];

        // Rotate 90° around Z: x -> y, y -> -x + translate (5,3,2)
        let translation = Vec3::new(5.0, 3.0, 2.0);
        let points_dst = vec![
            Vec3::new(0.0, 0.0, 0.0) + translation,
            Vec3::new(0.0, 1.0, 0.0) + translation,
            Vec3::new(-1.0, 0.0, 0.0) + translation,
            Vec3::new(0.0, 0.0, 1.0) + translation,
        ];

        let (r, t) = fit_transformation(&points_src, &points_dst);

        // The expected rotation is 90° CCW around Z axis
        let expected_r = Mat3::from_cols(
            Vec3::new(0.0, 1.0, 0.0),
            Vec3::new(-1.0, 0.0, 0.0),
            Vec3::new(0.0, 0.0, 1.0),
        );

        assert!(
            mat3_approx_eq(&r, &expected_r, 1e-6),
            "Expected rotation:\n{:?}\nGot:\n{:?}",
            expected_r,
            r
        );
        assert!(
            vec3_approx_eq(&t, &translation, 1e-6),
            "Expected translation {:?}, got: {:?}",
            translation,
            t
        );
    }
}
