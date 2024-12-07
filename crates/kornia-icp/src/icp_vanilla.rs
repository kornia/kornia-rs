use kiddo::float::distance::SquaredEuclidean;
use kiddo::float::kdtree::KdTree;

use crate::ops::{compute_point_to_point_error, fit_transformation};
use kornia_3d::{linalg::transform_points, pointcloud::PointCloud};

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

/// Iterative Closest Point cost function.
#[derive(Debug)]
pub enum IterativeClosestPointCostFunction {
    /// Point to point distance.
    PointToPoint,
    /// Point to plane distance.
    PointToPlane,
}

/// Transform a point cloud using a rotation and translation.
/// Iterative Closest Point (ICP) algorithm.
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
    mode: IterativeClosestPointCostFunction,
) -> Result<IterativeClosestPointResult, Box<dyn std::error::Error>> {
    // convert points to list of floats to avoid reallocating memory at each iteration
    let source_points = source
        .points()
        .iter()
        .map(|p| [p.x, p.y, p.z])
        .collect::<Vec<_>>();

    let target_points = target
        .points()
        .iter()
        .map(|p| [p.x, p.y, p.z])
        .collect::<Vec<_>>();

    // pre-allocate the matrices to avoid reallocating them at each iteration
    let mut transformed_points = vec![[0.0; 3]; source_points.len()];

    // build kdtree for target points to speed up the nearest neighbor search
    let mut kdtree: KdTree<f64, usize, 3, 32, u16> = KdTree::with_capacity(target_points.len());
    target_points.iter().enumerate().for_each(|(i, p)| {
        kdtree.add(p, i);
    });

    // initialize current source with the initial source point cloud
    let mut current_source = source_points;

    // initialize the result structure with identity rotation and translation
    let mut result = IterativeClosestPointResult {
        rotation: [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
        translation: [0.0, 0.0, 0.0],
        num_iterations: 0,
        last_error: 0.0,
    };

    // icp loop
    for i in 0..max_iterations {
        println!("iteration: {}", i);

        // find closest points between current source and target
        let closest_points = current_source
            .iter()
            .map(|p| {
                let nn = kdtree.nearest_one::<SquaredEuclidean>(p);
                &target_points[nn.item]
            })
            .collect::<Vec<_>>();

        //println!("current_source: {:?}", current_source);
        //println!("closest_points: {:?}", closest_points);
        //println!("########################");

        // compute transformation between current source and closest points
        fit_transformation(
            &current_source,
            &closest_points,
            &mut result.rotation,
            &mut result.translation,
        );

        // transform current source using the computed transformation
        transform_points(
            &current_source,
            &result.rotation,
            &result.translation,
            &mut transformed_points,
        );

        println!("transformed_points: {:?}", transformed_points);
        println!("########################");

        // compute error between transformed source and target
        let error = match mode {
            IterativeClosestPointCostFunction::PointToPoint => {
                compute_point_to_point_error(&transformed_points, &target_points)
            }
            // TODO: support point to plane distance
            IterativeClosestPointCostFunction::PointToPlane => {
                unimplemented!()
            }
        };

        current_source = transformed_points.clone();

        // update iteration count
        result.num_iterations += 1;
        result.last_error = error;

        // check convergence and exit if below tolerance
        if error < tolerance {
            break;
        }
    }

    Ok(result)
}
