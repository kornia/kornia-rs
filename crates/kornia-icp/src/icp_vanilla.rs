use core::f64;

use kiddo::immutable::float::kdtree::ImmutableKdTree;

use crate::ops::{find_correspondences, fit_transformation, update_transformation};
use kornia_3d::{linalg::transform_points3d, pointcloud::PointCloud};

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
        log::debug!("Iteration: {}", i);
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
        );

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
            log::debug!("ICP converged in {} iterations with error {}", i, rmse);
            result.rmse = rmse;
            break;
        }

        // update the result structure
        result.rmse = rmse;

        // swap current source with transformed points for the next iteration
        current_source = transformed_points;

        let elapsed = now.elapsed();
        log::debug!("elapsed: {:?}", elapsed);
    }

    Ok(result)
}

#[cfg(test)]
mod tests {

    use super::{icp_vanilla, ICPConvergenceCriteria};
    use kornia_3d::{
        linalg::transform_points3d, pointcloud::PointCloud,
        transforms::axis_angle_to_rotation_matrix,
    };

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

        println!("result: {:?}", result);

        // TODO: fixme
        // for i in 0..3 {
        //     assert_relative_eq!(result.translation[i], dst_t_src[i], epsilon = 1e-1);
        //     for j in 0..3 {
        //         // TODO: implement convert back to axis angle
        //         assert_relative_eq!(result.rotation[i][j], dst_r_src[i][j], epsilon = 1e-2);
        //     }
        // }

        Ok(())
    }
}
