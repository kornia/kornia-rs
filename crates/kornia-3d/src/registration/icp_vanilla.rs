#![allow(non_snake_case)]
#![allow(unused_imports)]
#![allow(clippy::needless_range_loop)]
use kiddo::immutable::float::kdtree::ImmutableKdTree;

use super::ops::{
    compute_point_to_plane_rmse, find_correspondences, find_correspondences_with_indices,
    fit_transformation, fit_transformation_point_to_plane, update_transformation,
};
use crate::{
    linalg::transform_points3d, normal_estimation::estimate_normals, pointcloud::PointCloud,
    
};

/// Result of the ICP algorithm.
///
/// The transformation is from the source to the target frame.
#[derive(Debug, Clone)]
pub struct ICPResult {
    /// Estimated rotation matrix.
    pub rotation: [[f64; 3]; 3],
    /// Estimated translation vector.
    pub translation: [f64; 3],
    /// The total number of iterations performed until convergence.
    pub num_iterations: usize,
    /// last computed RMSE.
    pub rmse: f64,
}

/// Structure to define the ICP parameters.
#[derive(Debug, Clone)]
pub struct ICPConvergenceCriteria {
    /// Maximum number of iterations to perform.
    pub max_iterations: usize,
    /// Convergence tolerance as the difference in RMSE between two consecutive iterations.
    pub tolerance: f64,
}

/// Iterative Closest Point (ICP) algorithm using point to point distance.
///
/// # Arguments
///
/// * `source` - Source point cloud.
/// * `target` - Target point cloud.
/// * `initial_rot` - Initial rotation matrix. This is the rotation from the source to the target frame.
/// * `initial_trans` - Initial translation vector. This is the translation from the source to the target frame.
/// * `criteria` - Convergence criteria.
///
/// # Returns
///
/// * `result` - Result of the ICP algorithm containing the rotation, translation, and number of iterations.
pub fn icp_vanilla(
    source: &PointCloud,
    target: &PointCloud,
    initial_rot: [[f64; 3]; 3],
    initial_trans: [f64; 3],
    criteria: ICPConvergenceCriteria,
) -> Result<ICPResult, Box<dyn std::error::Error>> {
    // initialize the result structure with the initial transformation given by the user
    let mut result = ICPResult {
        rotation: initial_rot,
        translation: initial_trans,
        num_iterations: 0,
        rmse: f64::INFINITY,
    };

    // build kdtree for target points to speed up the nearest neighbor search
    let kdtree: ImmutableKdTree<f64, u32, 3, 32> = ImmutableKdTree::new_from_slice(target.points());

    // perform transformation using the initial rotation and translation
    let mut transformed_points = vec![[0.0; 3]; source.points().len()];
    transform_points3d(
        source.points(),
        &result.rotation,
        &result.translation,
        &mut transformed_points,
    )?;

    // initialize current source with the initial source point cloud
    let mut current_source = transformed_points;

    // main icp loop
    for i in 0..criteria.max_iterations {
        // NOTE: for debugging purposes, we measure the time taken for each iteration
        log::debug!("Iteration: {i}");
        let now = std::time::Instant::now();

        // find closest points between current source and target
        let (current_source_match, current_target_match, distances) =
            find_correspondences(&current_source, target.points(), &kdtree);

        log::debug!(
            "Num correspondences: {}-{}",
            current_source_match.len(),
            current_target_match.len()
        );

        // compute transformation between current source and closest points
        let mut rr_delta = [[0.0; 3]; 3];
        let mut tt_delta = [0.0; 3];
        fit_transformation(
            &current_source_match,
            &current_target_match,
            &mut rr_delta,
            &mut tt_delta,
        )
        .ok_or("SVD failed in fit_transformation")?;

        // transform current source using the computed transformation
        let mut transformed_points = vec![[0.0; 3]; current_source.len()];
        transform_points3d(
            &current_source,
            &rr_delta,
            &tt_delta,
            &mut transformed_points,
        )?;

        // update the output transformation as
        // R_new = R_old * R_delta
        // t_new = t_old + t_delta
        update_transformation(
            &mut result.rotation,
            &mut result.translation,
            &rr_delta,
            &tt_delta,
        );

        // compute error between transformed source and target
        let rmse = (distances.iter().sum::<f64>() / distances.len() as f64).sqrt();

        // update the result structure
        result.num_iterations += 1;

        // check convergence and exit if below tolerance
        if (result.rmse - rmse).abs() < criteria.tolerance {
            log::debug!("ICP converged in {i} iterations with error {rmse}");
            result.rmse = rmse;
            break;
        }

        // update the result structure
        result.rmse = rmse;

        // swap current source with transformed points for the next iteration
        current_source = transformed_points;

        let elapsed = now.elapsed();
        log::debug!("elapsed: {elapsed:?}");
    }

    Ok(result)
}

/// Point-to-plane Iterative Closest Point (ICP) algorithm.
///
/// This variant uses the point-to-plane error metric, which converges faster
/// on planar surfaces. It requires normals for the target point cloud.
///
/// # Arguments
/// * `source` - Source point cloud.
/// * `target` - Target point cloud.
/// * `initial_rot` - Initial rotation matrix.
/// * `initial_trans` - Initial translation vector.
/// * `criteria` - Convergence criteria.
///
/// # Returns
/// An `ICPResult` containing the final transformation.
pub fn icp_point_to_plane(
    source: &PointCloud,
    target: &PointCloud,
    initial_rot: [[f64; 3]; 3],
    initial_trans: [f64; 3],
    criteria: ICPConvergenceCriteria,
) -> Result<ICPResult, Box<dyn std::error::Error>> {
    // --- Compute target normals using Phase 1 ---
    let target_with_normals = estimate_normals(target, 10)?;
    let target_normals = target_with_normals
        .normals()
        .ok_or("Failed to retrieve normals")?;

    // --- Build kd-tree for target points ---
    let kdtree: ImmutableKdTree<f64, u32, 3, 32> = ImmutableKdTree::new_from_slice(target.points());

    // --- Initialize result ---
    let mut result = ICPResult {
        rotation: initial_rot,
        translation: initial_trans,
        num_iterations: 0,
        rmse: f64::INFINITY,
    };

    // --- Apply initial transformation to source ---
    let mut current_source = vec![[0.0; 3]; source.points().len()];
    transform_points3d(
        source.points(),
        &result.rotation,
        &result.translation,
        &mut current_source,
    )?;

    // --- ICP loop ---
    for i in 0..criteria.max_iterations {
        let (src_matched, dst_matched, dst_indices, _) =
            find_correspondences_with_indices(&current_source, target.points(), &kdtree);

        let dst_normals_matched: Vec<[f64; 3]> = dst_indices
            .iter()
            .map(|&idx| target_normals[idx as usize])
            .collect();

        let identity_rot = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
        let identity_trans = [0.0, 0.0, 0.0];
        let (R_delta, t_delta) = fit_transformation_point_to_plane(
            &src_matched,
            &dst_matched,
            &dst_normals_matched,
            &identity_rot,
            &identity_trans,
        )?;

        let mut transformed_points = vec![[0.0; 3]; current_source.len()];
        transform_points3d(&current_source, &R_delta, &t_delta, &mut transformed_points)?;
        current_source = transformed_points;

        // Correct SE(3) update: R_new = R_old * R_delta, t_new = R_old * t_delta + t_old
        let mut new_rotation = [[0.0; 3]; 3];
        for i in 0..3 {
            for j in 0..3 {
                new_rotation[i][j] = 0.0;
                for k in 0..3 {
                    new_rotation[i][j] += result.rotation[i][k] * R_delta[k][j];
                }
            }
        }

        let mut new_translation = [0.0; 3];
        for i in 0..3 {
            new_translation[i] = result.translation[i];
            for j in 0..3 {
                new_translation[i] += result.rotation[i][j] * t_delta[j];
            }
        }

        result.rotation = new_rotation;
        result.translation = new_translation;

        // Transform the matched source points by the delta to get the new positions.
        let mut matched_src_transformed = vec![[0.0; 3]; src_matched.len()];
        transform_points3d(
            &src_matched,
            &R_delta,
            &t_delta,
            &mut matched_src_transformed,
        )?;

        // Compute RMSE using identity (because matched_src_transformed already includes the full transform).
        let identity_rot = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
        let identity_trans = [0.0, 0.0, 0.0];
        let rmse = compute_point_to_plane_rmse(
            &matched_src_transformed,
            &dst_matched,
            &dst_normals_matched,
            &identity_rot,
            &identity_trans,
        );

        result.num_iterations += 1;
        if (result.rmse - rmse).abs() < criteria.tolerance {
            result.rmse = rmse;
            break;
        }
        result.rmse = rmse;
    }

    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;

    use crate::transforms::axis_angle_to_rotation_matrix;
    use rand::RngExt;
    #[test]
    fn test_icp_vanilla() -> Result<(), Box<dyn std::error::Error>> {
        let num_points = 100;
        let points_src = (0..num_points)
            .map(|_| {
                [
                    rand::random::<f64>(),
                    rand::random::<f64>(),
                    rand::random::<f64>(),
                ]
            })
            .collect::<Vec<_>>();

        let dst_r_src = axis_angle_to_rotation_matrix(&[1.0, 0.0, 0.0], 0.1)?;
        let dst_t_src = [0.1, 0.1, 0.1];

        let mut points_dst = vec![[0.0; 3]; points_src.len()];
        transform_points3d(&points_src, &dst_r_src, &dst_t_src, &mut points_dst)?;

        let src_pcl = PointCloud::new(points_src, None, None);
        let dst_pcl = PointCloud::new(points_dst, None, None);

        let initial_rot = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
        let initial_trans = [0.0, 0.0, 0.0];

        let result = icp_vanilla(
            &src_pcl,
            &dst_pcl,
            initial_rot,
            initial_trans,
            ICPConvergenceCriteria {
                max_iterations: 100,
                tolerance: 1e-6,
            },
        )?;

        println!("result: {result:?}");

        // Compute angular rotation error
        // R_error = R_estimated^T * R_ground_truth
        let mut r_error = [[0.0; 3]; 3];
        for (i, r_error_row) in r_error.iter_mut().enumerate() {
            for (j, r_error_cell) in r_error_row.iter_mut().enumerate() {
                *r_error_cell = result.rotation[0][i] * dst_r_src[0][j]
                    + result.rotation[1][i] * dst_r_src[1][j]
                    + result.rotation[2][i] * dst_r_src[2][j];
            }
        }

        // Compute angle from rotation matrix: angle = acos((trace(R) - 1) / 2)
        let trace = r_error[0][0] + r_error[1][1] + r_error[2][2];
        let angular_error = ((trace - 1.0) / 2.0).clamp(-1.0, 1.0).acos();

        // Compute L2 translation error
        let translation_error = ((result.translation[0] - dst_t_src[0]).powi(2)
            + (result.translation[1] - dst_t_src[1]).powi(2)
            + (result.translation[2] - dst_t_src[2]).powi(2))
        .sqrt();

        // Assert using meaningful geometric metrics
        assert!(
            angular_error < 0.1,
            "Angular rotation error too large: {} rad (expected < 0.1 rad)",
            angular_error
        );

        assert!(
            translation_error < 0.1,
            "Translation error too large: {} (expected < 0.1)",
            translation_error
        );

        // Verify convergence
        assert!(
            result.rmse < 1e-8,
            "ICP did not converge to low error: RMSE = {}",
            result.rmse
        );

        Ok(())
    }

    // ------------------------------------------------------------------------
    // Helpers
    // ------------------------------------------------------------------------

    fn make_plane(side: usize) -> Vec<[f64; 3]> {
        let mut points = Vec::with_capacity(side * side);
        for i in 0..side {
            for j in 0..side {
                points.push([
                    i as f64 - side as f64 / 2.0,
                    j as f64 - side as f64 / 2.0,
                    0.0,
                ]);
            }
        }
        points
    }

    fn make_sphere(samples: usize) -> Vec<[f64; 3]> {
        let mut points = Vec::with_capacity(samples * samples);
        let step_theta = std::f64::consts::PI / (samples + 1) as f64;
        let step_phi = std::f64::consts::TAU / samples as f64;
        for i in 1..=samples {
            let theta = i as f64 * step_theta;
            for j in 0..samples {
                let phi = j as f64 * step_phi;
                points.push([
                    theta.sin() * phi.cos(),
                    theta.sin() * phi.sin(),
                    theta.cos(),
                ]);
            }
        }
        points
    }

    // ------------------------------------------------------------------------
    // Test: Point-to-plane converges in fewer iterations
    // ------------------------------------------------------------------------

    #[test]
    fn test_point_to_plane_icp_fewer_iterations() -> Result<(), Box<dyn std::error::Error>> {
        let side = 20;
        let source_points = make_plane(side);
        let source_pcl = PointCloud::new(source_points, None, None);

        // Use a modest rotation (0.1 rad) to keep the problem well-conditioned.
        let axis = [1.0, 0.0, 0.0];
        let angle = 0.1;
        let R_known = axis_angle_to_rotation_matrix(&axis, angle)?;
        let t_known = [0.1, 0.05, 0.1];

        let mut target_points = vec![[0.0; 3]; source_pcl.points().len()];
        transform_points3d(source_pcl.points(), &R_known, &t_known, &mut target_points)?;
        let target_pcl = PointCloud::new(target_points, None, None);

        let criteria = ICPConvergenceCriteria {
            max_iterations: 100,
            tolerance: 1e-6,
        };

        let init_rot = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
        let init_trans = [0.0, 0.0, 0.0];

        let res_ptpl =
            icp_point_to_plane(&source_pcl, &target_pcl, init_rot, init_trans, criteria)?;

        // 1. Check that point-to-plane converges within a reasonable number of iterations.
        assert!(
            res_ptpl.num_iterations < 15,
            "Point-to-plane should converge within 15 iterations, got {}",
            res_ptpl.num_iterations
        );

        // 2. Verify geometric accuracy (same pattern as test_icp_vanilla).
        let mut r_error = [[0.0; 3]; 3];
        for i in 0..3 {
            for j in 0..3 {
                r_error[i][j] = res_ptpl.rotation[0][i] * R_known[0][j]
                    + res_ptpl.rotation[1][i] * R_known[1][j]
                    + res_ptpl.rotation[2][i] * R_known[2][j];
            }
        }
        let trace = r_error[0][0] + r_error[1][1] + r_error[2][2];
        let angular_error = ((trace - 1.0) / 2.0).clamp(-1.0, 1.0).acos();

        // Only check the normal component (tz) – in-plane components are unobservable.
        let tz_error = (res_ptpl.translation[2] - t_known[2]).abs();
        assert!(
            tz_error < 0.06,
            "Normal translation error too large: {}",
            tz_error
        );

        // Also check RMSE is small (the objective is minimized).
        assert!(
            res_ptpl.rmse < 1e-4,
            "RMSE should be small: {}",
            res_ptpl.rmse
        );
        assert!(
            angular_error < 0.01,
            "Angular error too large: {} rad",
            angular_error
        );

        Ok(())
    }
    // ------------------------------------------------------------------------
    // Test: Flat plane correctness
    // ------------------------------------------------------------------------

    #[test]
    fn test_point_to_plane_icp_flat_plane() -> Result<(), Box<dyn std::error::Error>> {
        let side = 15;
        let source_points = make_plane(side);
        let source_pcl = PointCloud::new(source_points, None, None);

        let axis = [1.0, 0.0, 0.0];
        let angle = 0.1;
        let R_known = axis_angle_to_rotation_matrix(&axis, angle)?;
        let t_known = [0.1, 0.05, 0.1];

        let mut target_points = vec![[0.0; 3]; source_pcl.points().len()];
        transform_points3d(source_pcl.points(), &R_known, &t_known, &mut target_points)?;
        let target_pcl = PointCloud::new(target_points, None, None);

        let criteria = ICPConvergenceCriteria {
            max_iterations: 100,
            tolerance: 1e-6,
        };

        let init_rot = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
        let init_trans = [0.0, 0.0, 0.0];

        let res = icp_point_to_plane(&source_pcl, &target_pcl, init_rot, init_trans, criteria)?;

        assert!(res.rmse < 1e-3, "RMSE should be small: {}", res.rmse);
        assert!(
            res.num_iterations < 50,
            "Should converge in reasonable iterations: {}",
            res.num_iterations
        );

        Ok(())
    }

    // ------------------------------------------------------------------------
    // Test: Tilted plane correctness
    // ------------------------------------------------------------------------

    #[test]
    fn test_point_to_plane_icp_tilted_plane() -> Result<(), Box<dyn std::error::Error>> {
        let side = 15;
        let mut source_points = Vec::new();
        for i in 0..side {
            for j in 0..side {
                let x = i as f64 - side as f64 / 2.0;
                let y = j as f64 - side as f64 / 2.0;
                source_points.push([x, y, x + y]); // tilted: z = x + y
            }
        }
        let source_pcl = PointCloud::new(source_points, None, None);

        let axis = [0.0, 1.0, 0.0];
        let angle = 0.1;
        let R_known = axis_angle_to_rotation_matrix(&axis, angle)?;
        let t_known = [0.1, 0.05, 0.1];

        let mut target_points = vec![[0.0; 3]; source_pcl.points().len()];
        transform_points3d(source_pcl.points(), &R_known, &t_known, &mut target_points)?;
        let target_pcl = PointCloud::new(target_points, None, None);

        let criteria = ICPConvergenceCriteria {
            max_iterations: 100,
            tolerance: 1e-6,
        };

        let init_rot = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
        let init_trans = [0.0, 0.0, 0.0];

        let res = icp_point_to_plane(&source_pcl, &target_pcl, init_rot, init_trans, criteria)?;

        assert!(res.rmse < 1e-4, "RMSE should be small: {}", res.rmse);
        assert!(
            res.num_iterations < 30,
            "Should converge in reasonable iterations: {}",
            res.num_iterations
        );

        Ok(())
    }

    // ------------------------------------------------------------------------
    // Test: Noise robustness
    // ------------------------------------------------------------------------

    #[test]
    fn test_point_to_plane_icp_with_noise() -> Result<(), Box<dyn std::error::Error>> {
        let side = 15;
        let source_points = make_plane(side);
        let source_pcl = PointCloud::new(source_points, None, None);

        let axis = [1.0, 0.0, 0.0];
        let angle = 0.1;
        let R_known = axis_angle_to_rotation_matrix(&axis, angle)?;
        let t_known = [0.1, 0.05, 0.1];

        let mut target_points = vec![[0.0; 3]; source_pcl.points().len()];
        transform_points3d(source_pcl.points(), &R_known, &t_known, &mut target_points)?;

        // Add Gaussian noise
        let mut rng = rand::rng();
        for point in &mut target_points {
            point[0] += rng.random_range(-0.01..0.01);
            point[1] += rng.random_range(-0.01..0.01);
            point[2] += rng.random_range(-0.01..0.01);
        }

        let target_pcl = PointCloud::new(target_points, None, None);

        let criteria = ICPConvergenceCriteria {
            max_iterations: 100,
            tolerance: 1e-4,
        };

        let init_rot = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
        let init_trans = [0.0, 0.0, 0.0];

        let res = icp_point_to_plane(&source_pcl, &target_pcl, init_rot, init_trans, criteria)?;

        println!("Noisy iterations: {}", res.num_iterations);
        assert!(
            res.rmse < 0.1,
            "RMSE should be within tolerance: {}",
            res.rmse
        );

        Ok(())
    }

    // ------------------------------------------------------------------------
    // Test: Sphere (non‑planar surface)
    // ------------------------------------------------------------------------

    #[test]
    fn test_point_to_plane_icp_sphere() -> Result<(), Box<dyn std::error::Error>> {
        let samples = 20;
        let source_points = make_sphere(samples);
        let source_pcl = PointCloud::new(source_points, None, None);

        let axis = [1.0, 0.0, 0.0];
        let angle = 0.1;
        let R_known = axis_angle_to_rotation_matrix(&axis, angle)?;
        let t_known = [0.1, 0.0, 0.0];

        let mut target_points = vec![[0.0; 3]; source_pcl.points().len()];
        transform_points3d(source_pcl.points(), &R_known, &t_known, &mut target_points)?;
        let target_pcl = PointCloud::new(target_points, None, None);

        let criteria = ICPConvergenceCriteria {
            max_iterations: 100,
            tolerance: 1e-6,
        };

        let init_rot = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
        let init_trans = [0.0, 0.0, 0.0];

        let res = icp_point_to_plane(&source_pcl, &target_pcl, init_rot, init_trans, criteria)?;

        assert!(
            res.num_iterations < 30,
            "Should converge in reasonable iterations: {}",
            res.num_iterations
        );
        assert!(res.rmse < 1e-2, "Should achieve low RMSE: {}", res.rmse);

        Ok(())
    }

    // ------------------------------------------------------------------------
    // Test: Edge cases (small point cloud)
    // ------------------------------------------------------------------------

    #[test]
    fn test_point_to_plane_icp_edge_cases() -> Result<(), Box<dyn std::error::Error>> {
        let mut source_points = Vec::new();
        for i in 0..7 {
            for j in 0..7 {
                source_points.push([i as f64 - 3.0, j as f64 - 3.0, 0.0]);
            }
        }
        let source_pcl = PointCloud::new(source_points, None, None);

        let axis = [1.0, 0.0, 0.0];
        let angle = 0.1;
        let R_known = axis_angle_to_rotation_matrix(&axis, angle)?;
        let t_known = [0.01, 0.01, 0.0];

        let mut target_points = vec![[0.0; 3]; source_pcl.points().len()];
        transform_points3d(source_pcl.points(), &R_known, &t_known, &mut target_points)?;
        let target_pcl = PointCloud::new(target_points, None, None);

        let criteria = ICPConvergenceCriteria {
            max_iterations: 200,
            tolerance: 1e-6,
        };

        let init_rot = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
        let init_trans = [0.0, 0.0, 0.0];

        let res = icp_point_to_plane(&source_pcl, &target_pcl, init_rot, init_trans, criteria)?;

        // Compute angular error against ground truth (same pattern).
        let mut r_error = [[0.0; 3]; 3];
        for i in 0..3 {
            for j in 0..3 {
                r_error[i][j] = res.rotation[0][i] * R_known[0][j]
                    + res.rotation[1][i] * R_known[1][j]
                    + res.rotation[2][i] * R_known[2][j];
            }
        }
        let trace = r_error[0][0] + r_error[1][1] + r_error[2][2];
        let angular_error = ((trace - 1.0) / 2.0).clamp(-1.0, 1.0).acos();

        let translation_error = ((res.translation[0] - t_known[0]).powi(2)
            + (res.translation[1] - t_known[1]).powi(2)
            + (res.translation[2] - t_known[2]).powi(2))
        .sqrt();

        assert!(
            angular_error < 0.1,
            "Angular error too large: {} rad",
            angular_error
        );
        assert!(
            translation_error < 0.1,
            "Translation error too large: {}",
            translation_error
        );

        Ok(())
    }
    // ------------------------------------------------------------------------
    // Test: Compare iteration counts on sphere (non‑planar)
    // ------------------------------------------------------------------------

    #[test]
    fn test_point_to_plane_icp_fewer_iterations_sphere() -> Result<(), Box<dyn std::error::Error>> {
        let samples = 15;
        let source_points = make_sphere(samples);
        let source_pcl = PointCloud::new(source_points, None, None);

        let axis = [1.0, 0.0, 0.0];
        let angle = 0.5; // Increased from 0.1
        let R_known = axis_angle_to_rotation_matrix(&axis, angle)?;
        let t_known = [0.1, 0.05, 0.1];

        let mut target_points = vec![[0.0; 3]; source_pcl.points().len()];
        transform_points3d(source_pcl.points(), &R_known, &t_known, &mut target_points)?;
        let target_pcl = PointCloud::new(target_points, None, None);

        let criteria = ICPConvergenceCriteria {
            max_iterations: 100,
            tolerance: 1e-6,
        };

        let init_rot = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
        let init_trans = [0.0, 0.0, 0.0];

        let res_ptp = icp_vanilla(
            &source_pcl,
            &target_pcl,
            init_rot,
            init_trans,
            criteria.clone(),
        )?;
        let res_ptpl =
            icp_point_to_plane(&source_pcl, &target_pcl, init_rot, init_trans, criteria)?;

        println!(
            "Sphere – Point‑to‑point iterations: {}",
            res_ptp.num_iterations
        );
        println!(
            "Sphere – Point‑to‑plane iterations: {}",
            res_ptpl.num_iterations
        );

        assert!(
            res_ptpl.num_iterations < 50,
            "Point‑to‑plane should converge in reasonable iterations, got {}",
            res_ptpl.num_iterations
        );

        Ok(())
    }
}
