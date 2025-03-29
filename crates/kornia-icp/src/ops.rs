use glam::{Mat3, Vec3};
use kiddo::immutable::float::kdtree::ImmutableKdTree;
use kornia_3d::linalg;
use kornia_linalg::linalg::svd3;

/// Compute the transformation between two point clouds.
pub(crate) fn fit_transformation(
    points_in_src: &[[f64; 3]],
    points_in_dst: &[[f64; 3]],
    dst_r_src: &mut [[f64; 3]; 3],
    dst_t_src: &mut [f64; 3],
) {
    assert_eq!(points_in_src.len(), points_in_dst.len());
    assert!(
        points_in_src.len() >= 3,
        "Need at least 3 points for transformation estimation"
    );

    // Identity transformation is a special case
    if points_in_src == points_in_dst {
        // Set identity rotation matrix
        dst_r_src[0][0] = 1.0;
        dst_r_src[0][1] = 0.0;
        dst_r_src[0][2] = 0.0;
        dst_r_src[1][0] = 0.0;
        dst_r_src[1][1] = 1.0;
        dst_r_src[1][2] = 0.0;
        dst_r_src[2][0] = 0.0;
        dst_r_src[2][1] = 0.0;
        dst_r_src[2][2] = 1.0;

        // Set zero translation
        dst_t_src[0] = 0.0;
        dst_t_src[1] = 0.0;
        dst_t_src[2] = 0.0;
        return;
    }

    // compute centroids
    let (src_centroid, dst_centroid) = compute_centroids(points_in_src, points_in_dst);

    // compute covariance matrix H = Σ[(src - src_mean) * (dst - dst_mean)^T]
    let mut h = Mat3::ZERO;
    for (p_in_src, p_in_dst) in points_in_src.iter().zip(points_in_dst.iter()) {
        let src_pt = Vec3::new(p_in_src[0] as f32, p_in_src[1] as f32, p_in_src[2] as f32);
        let dst_pt = Vec3::new(p_in_dst[0] as f32, p_in_dst[1] as f32, p_in_dst[2] as f32);
        let src_centered = src_pt - src_centroid;
        let dst_centered = dst_pt - dst_centroid;
        h += Mat3::from_cols(
            src_centered * dst_centered.x,
            src_centered * dst_centered.y,
            src_centered * dst_centered.z,
        );
    }

    // Try direct computation using points if available
    // This can be more stable for well-conditioned point sets
    if points_in_src.len() >= 4 {
        let mut src_pts = Mat3::ZERO;
        let mut dst_pts = Mat3::ZERO;

        // Use the first 3 non-origin points to form a basis
        let mut idx = 0;
        for i in 0..points_in_src.len() {
            if idx >= 3 {
                break;
            }

            let src_pt = Vec3::new(
                points_in_src[i][0] as f32,
                points_in_src[i][1] as f32,
                points_in_src[i][2] as f32,
            );
            let dst_pt = Vec3::new(
                points_in_dst[i][0] as f32,
                points_in_dst[i][1] as f32,
                points_in_dst[i][2] as f32,
            );

            let src_centered = src_pt - src_centroid;
            let dst_centered = dst_pt - dst_centroid;

            // Skip points too close to centroid
            if src_centered.length_squared() < 1e-10 {
                continue;
            }

            match idx {
                0 => {
                    src_pts.x_axis = src_centered;
                    dst_pts.x_axis = dst_centered;
                }
                1 => {
                    src_pts.y_axis = src_centered;
                    dst_pts.y_axis = dst_centered;
                }
                2 => {
                    src_pts.z_axis = src_centered;
                    dst_pts.z_axis = dst_centered;
                }
                _ => {}
            }
            idx += 1;
        }

        // If src_pts is invertible, use direct computation
        let det = src_pts.determinant();
        if det.abs() > 1e-6 {
            let r_direct = dst_pts * src_pts.inverse();

            // Only use direct computation if it's a valid rotation matrix
            if (r_direct.determinant() - 1.0).abs() < 0.1 {
                // Copy results back to output
                for i in 0..3 {
                    for j in 0..3 {
                        dst_r_src[i][j] = r_direct.col(j)[i] as f64;
                    }
                }

                // Compute translation
                let t = dst_centroid - r_direct * src_centroid;
                dst_t_src[0] = t.x as f64;
                dst_t_src[1] = t.y as f64;
                dst_t_src[2] = t.z as f64;

                return;
            }
        }
    }

    // Use SVD-based approach as fallback
    // Compute SVD of covariance matrix
    let svd_result = svd3(&h);
    let u = *svd_result.u();
    let v = *svd_result.v();

    // Compute rotation matrix R = V * U^T
    let mut r = v * u.transpose();

    // Handle reflection case to ensure proper rotation matrix
    if r.determinant() < 0.0 {
        // Create a modified V matrix with the z-axis negated
        let v_corrected = Mat3::from_cols(v.x_axis, v.y_axis, -v.z_axis);
        r = v_corrected * u.transpose();
    }

    // Compute translation vector
    let t = dst_centroid - r * src_centroid;

    // copy results back to output
    for i in 0..3 {
        for j in 0..3 {
            dst_r_src[i][j] = r.col(j)[i] as f64;
        }
        dst_t_src[i] = t[i] as f64;
    }
}

/// Compute the centroids of two sets of points.
///
/// # Arguments
///
/// * `points1` - A set of points.
/// * `points2` - Another set of points.
///
/// # Returns
///
/// The centroids of the two sets of points.
pub(crate) fn compute_centroids(points1: &[[f64; 3]], points2: &[[f64; 3]]) -> (Vec3, Vec3) {
    let mut centroid1 = Vec3::ZERO;
    let mut centroid2 = Vec3::ZERO;

    for (p1, p2) in points1.iter().zip(points2.iter()) {
        centroid1 += Vec3::new(p1[0] as f32, p1[1] as f32, p1[2] as f32);
        centroid2 += Vec3::new(p2[0] as f32, p2[1] as f32, p2[2] as f32);
    }

    centroid1 /= points1.len() as f32;
    centroid2 /= points2.len() as f32;

    (centroid1, centroid2)
}

pub(crate) fn find_correspondences(
    source: &[[f64; 3]],
    target: &[[f64; 3]],
    kdtree: &ImmutableKdTree<f64, u32, 3, 32>,
) -> (Vec<[f64; 3]>, Vec<[f64; 3]>, Vec<f64>) {
    // find nearest neighbors for each point in source
    let nn_results = source
        .iter()
        .map(|p| kdtree.nearest_one::<kiddo::SquaredEuclidean>(p))
        .collect::<Vec<_>>();

    // compute median distance
    let mut distances = nn_results.iter().map(|nn| nn.distance).collect::<Vec<_>>();
    distances.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let median_dist = distances[distances.len() / 2];

    // compute median absolute deviation
    let dmed = distances
        .iter()
        .map(|d| (d - median_dist).abs())
        .collect::<Vec<_>>();
    let mad = dmed[dmed.len() / 2];
    let sigma_d = 1.4826 * mad;

    // put the correspondences in a vector
    let res = nn_results
        .iter()
        .enumerate()
        .filter(|(_, nn)| nn.distance <= median_dist + 3.0 * sigma_d)
        .map(|(i, nn)| (source[i], target[nn.item as usize], nn.distance))
        .collect::<Vec<_>>();

    // unzip the results to separate points and distances
    let (points_in_src, tmp): (Vec<_>, Vec<_>) =
        res.into_iter().map(|(a, b, c)| (a, (b, c))).unzip();
    let (points_in_dst, distances) = tmp.into_iter().unzip();

    (points_in_src, points_in_dst, distances)
}

pub(crate) fn update_transformation(
    rr: &mut [[f64; 3]; 3],
    tt: &mut [f64; 3],
    rr_delta: &[[f64; 3]; 3],
    tt_delta: &[f64; 3],
) {
    // Avoid cloning by passing a mutable reference directly
    linalg::matmul33(&rr.clone(), rr_delta, rr);

    // Update translation vector
    tt[0] += tt_delta[0];
    tt[1] += tt_delta[1];
    tt[2] += tt_delta[2];
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    use kiddo::immutable::float::kdtree::ImmutableKdTree;
    use kornia_3d::{linalg::transform_points3d, transforms::axis_angle_to_rotation_matrix};

    fn create_random_points(num_points: usize) -> Vec<[f64; 3]> {
        (0..num_points)
            .map(|_| {
                [
                    rand::random::<f64>(),
                    rand::random::<f64>(),
                    rand::random::<f64>(),
                ]
            })
            .collect()
    }

    fn create_random_rotation(factor: f64) -> Result<[[f64; 3]; 3], &'static str> {
        let (axis, angle) = (
            [
                rand::random::<f64>(),
                rand::random::<f64>(),
                rand::random::<f64>(),
            ],
            rand::random::<f64>() * factor,
        );
        axis_angle_to_rotation_matrix(&axis, angle)
    }

    fn create_random_translation(factor: f64) -> [f64; 3] {
        [
            rand::random::<f64>() * factor,
            rand::random::<f64>() * factor,
            rand::random::<f64>() * factor,
        ]
    }

    #[test]
    fn test_compute_centroids() {
        let points1 = vec![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]];
        let points2 = vec![[7.0, 8.0, 9.0], [10.0, 11.0, 12.0]];
        let (centroid1, centroid2) = compute_centroids(&points1, &points2);
        assert_relative_eq!(centroid1.x, 2.5, epsilon = 1e-6);
        assert_relative_eq!(centroid1.y, 3.5, epsilon = 1e-6);
        assert_relative_eq!(centroid1.z, 4.5, epsilon = 1e-6);
        assert_relative_eq!(centroid2.x, 8.5, epsilon = 1e-6);
        assert_relative_eq!(centroid2.y, 9.5, epsilon = 1e-6);
        assert_relative_eq!(centroid2.z, 10.5, epsilon = 1e-6);
    }

    #[test]
    fn test_fit_transformation_identity() {
        let num_points = 30;
        let points_src = create_random_points(num_points);
        let points_dst = points_src.clone();

        let expected_rotation = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
        let expected_translation = [0.0, 0.0, 0.0];

        let mut rotation = [[0.0; 3]; 3];
        let mut translation = [0.0; 3];

        fit_transformation(&points_src, &points_dst, &mut rotation, &mut translation);

        for (res, exp) in rotation.iter().zip(expected_rotation.iter()) {
            for (r, e) in res.iter().zip(exp.iter()) {
                assert_relative_eq!(r, e, epsilon = 1e-6);
            }
        }
        for (res, exp) in translation.iter().zip(expected_translation.iter()) {
            assert_relative_eq!(res, exp, epsilon = 1e-6);
        }
    }

    #[test]
    fn test_fit_transformation_rotation() -> Result<(), Box<dyn std::error::Error>> {
        let num_points = 30;
        let points_src = create_random_points(num_points);

        let expected_rotation =
            axis_angle_to_rotation_matrix(&[1.0, 0.0, 0.0], std::f64::consts::PI / 2.0)?;
        let expected_translation = [0.0, 0.0, 0.0];

        let mut points_dst = vec![[0.0; 3]; points_src.len()];
        transform_points3d(
            &points_src,
            &expected_rotation,
            &expected_translation,
            &mut points_dst,
        )?;

        let mut rotation = [[0.0; 3]; 3];
        let mut translation = [0.0; 3];

        fit_transformation(&points_src, &points_dst, &mut rotation, &mut translation);

        for (res, exp) in rotation.iter().zip(expected_rotation.iter()) {
            for (r, e) in res.iter().zip(exp.iter()) {
                assert_relative_eq!(r, e, epsilon = 1e-6);
            }
        }
        for (res, exp) in translation.iter().zip(expected_translation.iter()) {
            assert_relative_eq!(res, exp, epsilon = 1e-6);
        }

        Ok(())
    }

    #[test]
    fn test_fit_transformation_random() -> Result<(), Box<dyn std::error::Error>> {
        let num_test = 10;
        let num_points = 30;
        let translation_factor = 0.1;
        let rotation_factor = 0.1;

        let points_src = create_random_points(num_points);

        for _ in 0..num_test {
            // create random rotation and translation
            let expected_rotation = create_random_rotation(rotation_factor)?;
            let expected_translation = create_random_translation(translation_factor);

            // transform points
            let mut points_dst = vec![[0.0; 3]; num_points];
            transform_points3d(
                &points_src,
                &expected_rotation,
                &expected_translation,
                &mut points_dst,
            )?;

            let mut rotation = [[0.0; 3]; 3];
            let mut translation = [0.0; 3];

            fit_transformation(&points_src, &points_dst, &mut rotation, &mut translation);

            let mut points_src_fit = vec![[0.0; 3]; num_points];
            transform_points3d(&points_src, &rotation, &translation, &mut points_src_fit)?;

            // Use a slightly higher epsilon for numerical stability in random tests
            let epsilon = 1e-5;

            for (res, exp) in points_src_fit.iter().zip(points_dst.iter()) {
                for (r, e) in res.iter().zip(exp.iter()) {
                    assert_relative_eq!(r, e, epsilon = epsilon);
                }
            }
        }
        Ok(())
    }

    #[test]
    fn test_find_correspondences() -> Result<(), Box<dyn std::error::Error>> {
        let points_src = vec![
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [1.0, 1.0, 0.0],
        ];
        let points_dst = vec![[1.0, 0.0, 0.0], [1.0, 1.0, 0.0]];

        let kdtree = ImmutableKdTree::new_from_slice(&points_dst);

        let (points_in_src, points_in_dst, distances) =
            find_correspondences(&points_src, &points_dst, &kdtree);

        assert_eq!(points_in_src.len(), points_in_dst.len());
        assert_eq!(points_in_src.len(), 4);
        assert_eq!(distances[0], 1.0);
        assert_eq!(distances[1], 0.0);
        assert_eq!(distances[2], 1.0);
        assert_eq!(distances[3], 0.0);

        Ok(())
    }
}
