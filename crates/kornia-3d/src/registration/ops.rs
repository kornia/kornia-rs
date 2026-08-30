use crate::linalg;
use faer::prelude::Solve;
use kiddo::immutable::float::kdtree::ImmutableKdTree;
use kornia_algebra::{Mat3F64, Vec3F64};
/// Errors that can occur during ICP registration.
#[derive(Debug, thiserror::Error)]
pub enum IcpError {
    #[error("LM failed to find a step that improves RMSE")]
    LmFailed,

    #[error("Mismatched correspondence lengths: expected {expected}, got {got} for {name}")]
    MismatchedLengths {
        name: &'static str,
        expected: usize,
        got: usize,
    },

    #[error("No correspondences found between source and target")]
    EmptyCorrespondences,

    #[error("Rotation matrix construction failed: {0}")]
    RotationFailed(String),
}

/// Compute the transformation between two point clouds.
pub(crate) fn fit_transformation(
    points_in_src: &[[f64; 3]],
    points_in_dst: &[[f64; 3]],
    dst_r_src: &mut [[f64; 3]; 3],
    dst_t_src: &mut [f64; 3],
) -> Option<()> {
    assert_eq!(points_in_src.len(), points_in_dst.len());

    // compute centroids
    let (src_centroid, dst_centroid) = compute_centroids(points_in_src, points_in_dst);

    // compute covariance matrix
    let mut hh = faer::Mat::<f64>::zeros(3, 3);
    for (p_in_src, p_in_dst) in points_in_src.iter().zip(points_in_dst.iter()) {
        let p_src = faer::col![p_in_src[0], p_in_src[1], p_in_src[2]] - &src_centroid;
        let p_dst = faer::col![p_in_dst[0], p_in_dst[1], p_in_dst[2]] - &dst_centroid;
        hh += p_src * p_dst.transpose();
    }

    // solve the linear system H * x = 0 to find the rotation
    let svd = hh.svd().ok()?;
    let (u_t, v) = (svd.U().transpose(), svd.V());

    // compute rotation matrix R = V * U^T
    let mut rr = v * u_t;

    // fix the determinant of R in case it is negative as it's a reflection matrix
    if rr.determinant() < 0.0 {
        log::warn!("WARNING: det(R) < 0.0, fixing it...");
        let v_neg = {
            let mut v_neg = v.to_owned();
            v_neg.col_mut(2).copy_from(-v.col(2));
            v_neg
        };
        // TODO: improve performance by using matmul33
        faer::linalg::matmul::matmul(
            &mut rr,
            faer::Accum::Replace,
            &v_neg,
            u_t,
            1.0_f64,
            faer::Par::Seq,
        );
    }

    // compute translation vector t = C_dst - R * C_src
    let t = dst_centroid - &rr * src_centroid;

    // copy results back to output
    #[allow(clippy::needless_range_loop)]
    for i in 0..3 {
        for j in 0..3 {
            dst_r_src[i][j] = rr[(i, j)];
        }
        dst_t_src[i] = t[i];
    }
    Some(())
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
pub(crate) fn compute_centroids(
    points1: &[[f64; 3]],
    points2: &[[f64; 3]],
) -> (faer::Col<f64>, faer::Col<f64>) {
    let mut centroid1 = faer::Col::zeros(3);
    let mut centroid2 = faer::Col::zeros(3);

    for (p1, p2) in points1.iter().zip(points2.iter()) {
        centroid1 += faer::col![p1[0], p1[1], p1[2]];
        centroid2 += faer::col![p2[0], p2[1], p2[2]];
    }

    centroid1 /= points1.len() as f64;
    centroid2 /= points2.len() as f64;

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

    let mut distances = nn_results.iter().map(|nn| nn.distance).collect::<Vec<_>>();
    if distances.is_empty() {
        return (Vec::new(), Vec::new(), Vec::new());
    }

    let mid_dist = distances.len() / 2;
    distances.select_nth_unstable_by(mid_dist, |a, b| a.total_cmp(b));
    let median_dist = distances[mid_dist];

    let mut dmed = nn_results
        .iter()
        .map(|nn| (nn.distance - median_dist).abs())
        .collect::<Vec<_>>();

    let mid_mad = dmed.len() / 2;
    dmed.select_nth_unstable_by(mid_mad, |a, b| a.total_cmp(b));
    let mad = dmed[mid_mad];

    let sigma_d = 1.4826 * mad;
    let threshold = median_dist + 3.0 * sigma_d;

    let res = nn_results
        .iter()
        .enumerate()
        .filter(|(_, nn)| nn.distance <= threshold)
        .map(|(i, nn)| (source[i], target[nn.item as usize], nn.distance))
        .collect::<Vec<_>>();

    let mut points_in_src = Vec::with_capacity(res.len());
    let mut points_in_dst = Vec::with_capacity(res.len());
    let mut distances_vec = Vec::with_capacity(res.len());

    for (src, dst, dist) in res {
        points_in_src.push(src);
        points_in_dst.push(dst);
        distances_vec.push(dist);
    }

    (points_in_src, points_in_dst, distances_vec)
}

pub(crate) fn update_transformation(
    rr: &mut [[f64; 3]; 3],
    tt: &mut [f64; 3],
    rr_delta: &[[f64; 3]; 3],
    tt_delta: &[f64; 3],
) {
    let r_old = *rr;
    let t_old = *tt;

    // LEFT composition: R_new = R_delta * R_old
    linalg::matmul33(rr_delta, &r_old, rr);

    // t_new = R_delta * t_old + t_delta
    tt[0] = rr_delta[0][0] * t_old[0]
        + rr_delta[0][1] * t_old[1]
        + rr_delta[0][2] * t_old[2]
        + tt_delta[0];
    tt[1] = rr_delta[1][0] * t_old[0]
        + rr_delta[1][1] * t_old[1]
        + rr_delta[1][2] * t_old[2]
        + tt_delta[1];
    tt[2] = rr_delta[2][0] * t_old[0]
        + rr_delta[2][1] * t_old[1]
        + rr_delta[2][2] * t_old[2]
        + tt_delta[2];
}

// ============================================================================
// NEW: Point-to-plane ICP helper functions
// ============================================================================

/// Find correspondences with indices (for point-to-plane ICP).
///
/// Identical to `find_correspondences` but also returns the indices of the
/// matched target points, needed for looking up normals.
type Correspondences = (Vec<[f64; 3]>, Vec<[f64; 3]>, Vec<u32>, Vec<f64>);
pub(crate) fn find_correspondences_with_indices(
    source: &[[f64; 3]],
    target: &[[f64; 3]],
    kdtree: &ImmutableKdTree<f64, u32, 3, 32>,
) -> Correspondences {
    let nn_results = source
        .iter()
        .map(|p| kdtree.nearest_one::<kiddo::SquaredEuclidean>(p))
        .collect::<Vec<_>>();

    let mut distances: Vec<f64> = nn_results.iter().map(|nn| nn.distance).collect();
    if distances.is_empty() {
        return Correspondences::default();
    }

    let mid_dist = distances.len() / 2;
    distances.select_nth_unstable_by(mid_dist, |a, b| a.total_cmp(b));
    let median_dist = distances[mid_dist];

    let mut dmed = nn_results
        .iter()
        .map(|nn| (nn.distance - median_dist).abs())
        .collect::<Vec<_>>();
    let mid_mad = dmed.len() / 2;
    dmed.select_nth_unstable_by(mid_mad, |a, b| a.total_cmp(b));
    let mad = dmed[mid_mad];

    let sigma_d = 1.4826 * mad;
    let threshold = median_dist + 3.0 * sigma_d;

    // Single pass: collect filtered correspondences, then push into four vectors.
    let mut src_matched = Vec::new();
    let mut dst_matched = Vec::new();
    let mut dst_indices = Vec::new();
    let mut distances_out = Vec::new();

    for (i, nn) in nn_results.iter().enumerate() {
        if nn.distance <= threshold {
            src_matched.push(source[i]);
            dst_matched.push(target[nn.item as usize]);
            dst_indices.push(nn.item);
            distances_out.push(nn.distance);
        }
    }

    (src_matched, dst_matched, dst_indices, distances_out)
}

/// Compute point-to-plane RMSE.
pub(crate) fn compute_point_to_plane_rmse(
    src: &[[f64; 3]],
    dst: &[[f64; 3]],
    normals: &[[f64; 3]],
    rot: &[[f64; 3]; 3],
    trans: &[f64; 3],
) -> f64 {
    if src.is_empty() {
        return f64::INFINITY;
    }

    let rot_mat = Mat3F64::from_cols_array(&[
        rot[0][0], rot[1][0], rot[2][0], rot[0][1], rot[1][1], rot[2][1], rot[0][2], rot[1][2],
        rot[2][2],
    ]);
    let trans_vec = Vec3F64::new(trans[0], trans[1], trans[2]);

    let sum_sq: f64 = src
        .iter()
        .zip(dst.iter())
        .zip(normals.iter())
        .map(|((p, q), n)| {
            let p_vec = Vec3F64::new(p[0], p[1], p[2]);
            let q_vec = Vec3F64::new(q[0], q[1], q[2]);
            let n_vec = Vec3F64::new(n[0], n[1], n[2]);

            let p_trans = rot_mat * p_vec + trans_vec;
            let diff = p_trans - q_vec;
            let proj = diff.dot(n_vec);
            proj * proj
        })
        .sum();

    (sum_sq / src.len() as f64).sqrt()
}

/// Estimate incremental transformation using point-to-plane error metric.
/// Solves the linearized least squares problem:
/// hessian · x = gradient
/// where hessian = jacᵀ · jac and gradient = jacᵀ · residuals.
///Returns (rot_delta, trans_delta).
pub(crate) fn fit_transformation_point_to_plane(
    src_points: &[[f64; 3]],
    dst_points: &[[f64; 3]],
    dst_normals: &[[f64; 3]],
    rot: &[[f64; 3]; 3],
    trans: &[f64; 3],
) -> Result<([[f64; 3]; 3], [f64; 3]), IcpError> {
    // Validate slice lengths so we don't panic
    let m = src_points.len();
    if dst_points.len() != m {
        return Err(IcpError::MismatchedLengths {
            name: "dst_points",
            expected: m,
            got: dst_points.len(),
        });
    }
    if dst_normals.len() != m {
        return Err(IcpError::MismatchedLengths {
            name: "dst_normals",
            expected: m,
            got: dst_normals.len(),
        });
    }

    if m == 0 {
        return Ok((
            [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
            [0.0, 0.0, 0.0],
        ));
    }

    let mut jac = faer::Mat::<f64>::zeros(m, 6);
    let mut residuals = faer::Mat::<f64>::zeros(m, 1);

    for i in 0..m {
        let p = src_points[i];
        let q = dst_points[i];
        let n = dst_normals[i];

        // Compute rot * p
        let rot_p = [
            rot[0][0] * p[0] + rot[0][1] * p[1] + rot[0][2] * p[2],
            rot[1][0] * p[0] + rot[1][1] * p[1] + rot[1][2] * p[2],
            rot[2][0] * p[0] + rot[2][1] * p[1] + rot[2][2] * p[2],
        ];

        // Full transformed point
        let p_trans = [
            rot_p[0] + trans[0],
            rot_p[1] + trans[1],
            rot_p[2] + trans[2],
        ];

        // Residual
        let residual =
            (p_trans[0] - q[0]) * n[0] + (p_trans[1] - q[1]) * n[1] + (p_trans[2] - q[2]) * n[2];

        // Jacobian rotation part: rot_p × n
        let cross = [
            rot_p[1] * n[2] - rot_p[2] * n[1],
            rot_p[2] * n[0] - rot_p[0] * n[2],
            rot_p[0] * n[1] - rot_p[1] * n[0],
        ];

        jac[(i, 0)] = cross[0];
        jac[(i, 1)] = cross[1];
        jac[(i, 2)] = cross[2];
        jac[(i, 3)] = n[0];
        jac[(i, 4)] = n[1];
        jac[(i, 5)] = n[2];

        residuals[(i, 0)] = -residual;
    }

    // --- Levenberg-Marquardt damping ---
    let damping_factor = 1e-6;
    let hessian = jac.transpose() * &jac;
    let gradient = jac.transpose() * &residuals;

    let mut diag_sum = 0.0;
    for i in 0..6 {
        diag_sum += hessian[(i, i)];
    }
    let mut lambda = damping_factor * diag_sum / 6.0;

    // Compute current_rmse ONCE, outside the loop
    let current_rmse = compute_point_to_plane_rmse(src_points, dst_points, dst_normals, rot, trans);

    let mut x = None;

    for _ in 0..15 {
        let mut hessian_damped = hessian.clone();
        for i in 0..6 {
            // Absolute damping (not relative to H_ii)
            hessian_damped[(i, i)] += lambda;
        }

        let lu = hessian_damped.partial_piv_lu();
        let sol = lu.solve(&gradient);

        let all_finite = (0..6).all(|i| sol[(i, 0)].is_finite());
        if !all_finite {
            lambda *= 2.0;
            continue;
        }

        // Convert to rotation and translation delta
        let (rot_delta_tmp, trans_delta_tmp) = solution_to_delta(&sol)?;

        // Test candidate using update_transformation (same as icp_vanilla)
        let mut rot_tmp = *rot;
        let mut trans_tmp = *trans;
        update_transformation(
            &mut rot_tmp,
            &mut trans_tmp,
            &rot_delta_tmp,
            &trans_delta_tmp,
        );

        let new_rmse =
            compute_point_to_plane_rmse(src_points, dst_points, dst_normals, &rot_tmp, &trans_tmp);

        // Accept if objective improved (absolute epsilon for numerical noise)
        if new_rmse <= current_rmse + 1e-12 {
            x = Some(sol);
            break;
        }

        lambda *= 2.0;
    }

    let x = x.ok_or(IcpError::LmFailed)?;
    let (rot_delta, trans_delta) = solution_to_delta(&x)?;

    Ok((rot_delta, trans_delta))
}
/// Convert a 6-DOF LM solution [α, β, γ, tx, ty, tz] to (R_delta, t_delta).
fn solution_to_delta(sol: &faer::Mat<f64>) -> Result<([[f64; 3]; 3], [f64; 3]), IcpError> {
    let alpha = sol[(0, 0)];
    let beta = sol[(1, 0)];
    let gamma = sol[(2, 0)];
    let tx = sol[(3, 0)];
    let ty = sol[(4, 0)];
    let tz = sol[(5, 0)];

    let theta = (alpha * alpha + beta * beta + gamma * gamma).sqrt();
    let rot_delta = if theta < 1e-12 {
        [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
    } else {
        let axis = [alpha / theta, beta / theta, gamma / theta];
        crate::transforms::axis_angle_to_rotation_matrix(&axis, theta)
            .map_err(|e| IcpError::RotationFailed(e.to_string()))?
    };
    let trans_delta = [tx, ty, tz];
    Ok((rot_delta, trans_delta))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{linalg::transform_points3d, transforms::axis_angle_to_rotation_matrix};
    use approx::assert_relative_eq;
    use kiddo::immutable::float::kdtree::ImmutableKdTree;

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
        let c1: Vec<f64> = centroid1.as_ref().iter().copied().collect();
        let c2: Vec<f64> = centroid2.as_ref().iter().copied().collect();
        assert_eq!(c1, vec![2.5, 3.5, 4.5]);
        assert_eq!(c2, vec![8.5, 9.5, 10.5]);
    }

    #[test]
    fn test_fit_transformation_identity() -> Result<(), &'static str> {
        let num_points = 30;
        let points_src = create_random_points(num_points);
        let points_dst = points_src.clone();

        let expected_rotation = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
        let expected_translation = [0.0, 0.0, 0.0];

        let mut rotation = [[0.0; 3]; 3];
        let mut translation = [0.0; 3];

        fit_transformation(&points_src, &points_dst, &mut rotation, &mut translation)
            .ok_or("SVD failed")?;

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

        fit_transformation(&points_src, &points_dst, &mut rotation, &mut translation)
            .ok_or("SVD failed")?;

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

            fit_transformation(&points_src, &points_dst, &mut rotation, &mut translation)
                .ok_or("SVD failed")?;

            let mut points_src_fit = vec![[0.0; 3]; num_points];
            transform_points3d(&points_src, &rotation, &translation, &mut points_src_fit)?;

            for (res, exp) in points_src_fit.iter().zip(points_dst.iter()) {
                for (r, e) in res.iter().zip(exp.iter()) {
                    assert_relative_eq!(r, e, epsilon = 1e-6);
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
