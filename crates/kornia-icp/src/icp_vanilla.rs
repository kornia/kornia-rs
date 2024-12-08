use kiddo::float::kdtree::KdTree;

use crate::ops::{
    compute_point_to_point_error, find_correspondences, fit_transformation, update_transformation,
};
use kornia_3d::{linalg::transform_points3d, pointcloud::PointCloud};

/// Result of the ICP algorithm.
#[derive(Debug)]
pub struct IterativeClosestPointResult {
    /// Rotation matrix.
    pub rotation: [[f64; 3]; 3],
    /// Translation vector.
    pub translation: [f64; 3],
    /// Number of iterations.
    pub num_iterations: usize,
    /// last computed error
    pub last_error: f64,
}

/// Iterative Closest Point (ICP) algorithm using point to point distance.
///
/// # Arguments
///
/// * `source` - Source point cloud.
/// * `target` - Target point cloud.
/// * `max_iterations` - Maximum number of iterations.
/// * `tolerance` - Convergence tolerance.
///
/// # Returns
///
/// * `result` - Result of the ICP algorithm containing the rotation, translation, and number of iterations.
pub fn icp_vanilla(
    source: &PointCloud,
    target: &PointCloud,
    max_iterations: usize,
    tolerance: f64,
) -> Result<IterativeClosestPointResult, Box<dyn std::error::Error>> {
    // build kdtree for target points to speed up the nearest neighbor search
    let mut kdtree: KdTree<f64, usize, 3, 32, u16> = KdTree::with_capacity(target.len());
    target.points().iter().enumerate().for_each(|(i, p)| {
        kdtree.add(p, i);
    });

    // initialize current source with the initial source point cloud
    let mut current_source = source.points().clone();

    // initialize the result structure with identity rotation and translation
    let mut result = IterativeClosestPointResult {
        rotation: [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
        translation: [0.0, 0.0, 0.0],
        num_iterations: 0,
        last_error: 0.0,
    };

    // main icp loop
    for i in 0..max_iterations {
        // find closest points between current source and target
        let (current_source_match, current_target_match, distances) =
            find_correspondences(&current_source, target.points(), &kdtree);

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
        );

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
        let error = distances.iter().sum::<f64>() / distances.len() as f64;

        // update iteration count
        result.num_iterations += 1;
        result.last_error = error;

        // check convergence and exit if below tolerance
        if error < tolerance {
            log::info!("ICP converged in {} iterations with error {}", i, error);
            break;
        }

        // swap current source with transformed points for the next iteration
        current_source = transformed_points;
    }

    Ok(result)
}

#[cfg(test)]
mod tests {

    use super::icp_vanilla;
    use approx::assert_relative_eq;
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
        transform_points3d(&points_src, &dst_r_src, &dst_t_src, &mut points_dst);

        let src_pcl = PointCloud::new(points_src, None, None);
        let dst_pcl = PointCloud::new(points_dst, None, None);

        let result = icp_vanilla(&src_pcl, &dst_pcl, 100, 1e-6)?;

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
