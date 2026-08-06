//! Normal estimation for point clouds using Principal Component Analysis (PCA).
//!
//! This module provides functions to estimate surface normals from a point cloud
//! by analyzing the local neighborhood of each point via PCA on the covariance
//! matrix of the k-nearest neighbors.

use crate::pointcloud::PointCloud;
use faer::{Col, Mat};
use kiddo::immutable::float::kdtree::ImmutableKdTree;
use std::num::NonZeroUsize;
use thiserror::Error;

/// Errors that can occur during normal estimation.
#[derive(Debug, Error)]
pub enum NormalEstimationError {
    /// The point cloud is empty.
    #[error("Point cloud is empty")]
    EmptyPointCloud,

    /// The number of neighbors `k` is invalid.
    #[error("k must be between 2 and {max}, got {got}")]
    InvalidNeighborCount {
        /// Maximum allowed value.
        max: usize,
        /// Actual value provided.
        got: usize,
    },

    /// Singular Value Decomposition failed.
    #[error("SVD computation failed: {0}")]
    SvdFailed(String),

    /// Neighborhood is degenerate (all points identical or collinear).
    #[error("Degenerate neighborhood at point {index}: cannot estimate a unique normal")]
    DegenerateNeighborhood {
        /// Index of the point with the degenerate neighborhood.
        index: usize,
    },
}

/// Estimate surface normals using PCA.
///
/// For each point in the cloud, this function finds its `k` nearest neighbors,
/// computes the covariance matrix of the local neighborhood, and extracts the
/// eigenvector corresponding to the smallest eigenvalue as the normal.
///
/// # Arguments
/// * `cloud` - The input point cloud. Must have at least `k` points.
/// * `k` - The number of nearest neighbors to consider. Must be >= 2 and <= point count.
///
/// # Returns
/// A new `PointCloud` with the same points and colors as the input, and normals filled.
///
/// # Errors
/// Returns a `NormalEstimationError` if:
/// * The point cloud is empty.
/// * `k` is out of bounds.
/// * SVD fails (numerical issue).
/// * A local neighborhood is degenerate (all points identical).
///
/// # Performance
/// The function builds a k-d tree (O(N log N)) and then processes each point
/// with O(k log N) nearest-neighbor queries. The covariance and SVD are O(k + 1)
/// per point. For typical point clouds (N < 100k, k < 30), this is efficient.
pub fn estimate_normals(cloud: &PointCloud, k: usize) -> Result<PointCloud, NormalEstimationError> {
    let points = cloud.points();
    let n_points = points.len();

    if n_points == 0 {
        return Err(NormalEstimationError::EmptyPointCloud);
    }
    if k < 2 || k > n_points {
        return Err(NormalEstimationError::InvalidNeighborCount {
            max: n_points,
            got: k,
        });
    }

    let kdtree: ImmutableKdTree<f64, u32, 3, 32> = ImmutableKdTree::new_from_slice(points);

    let mut normals = Vec::with_capacity(n_points);

    for i in 0..n_points {
        let query = points[i];

        let neighbors =
            kdtree.nearest_n::<kiddo::SquaredEuclidean>(&query, NonZeroUsize::new(k).unwrap());

        let mut neighbor_points = Vec::with_capacity(k);
        for neighbour in &neighbors {
            let idx = neighbour.item as usize;
            neighbor_points.push(points[idx]);
        }
        let neighbor_count = neighbor_points.len();

        if neighbor_count < 2 {
            return Err(NormalEstimationError::DegenerateNeighborhood { index: i });
        }

        let mut mean = Col::zeros(3);
        for p in &neighbor_points {
            mean += faer::col![p[0], p[1], p[2]];
        }
        mean /= neighbor_count as f64;

        let mut cov = Mat::<f64>::zeros(3, 3);
        for p in &neighbor_points {
            let centered = faer::col![p[0], p[1], p[2]] - &mean;
            cov += centered.clone() * centered.transpose();
        }
        cov /= neighbor_count as f64;

        let svd = cov
            .svd()
            .map_err(|e| NormalEstimationError::SvdFailed(format!("{:?}", e)))?;

        let s = svd.S();
        let v = svd.V();

        // Find the minimum singular value and its index.
        // This is the normal direction.
        let mut min_idx = 0;
        let mut min_val = s[0];
        for j in 1..3 {
            if s[j] < min_val {
                min_val = s[j];
                min_idx = j;
            }
        }

        // Degeneracy check: the neighborhood must span a plane.
        // If the second largest singular value is near zero, the points are collinear.
        // If the largest singular value is near zero, all points are identical.
        // In both cases, a unique normal cannot be estimated.
        if s[1] < 1e-12 {
            return Err(NormalEstimationError::DegenerateNeighborhood { index: i });
        }

        let normal = [v[(0, min_idx)], v[(1, min_idx)], v[(2, min_idx)]];

        let len = (normal[0] * normal[0] + normal[1] * normal[1] + normal[2] * normal[2]).sqrt();
        if len > 1e-12 {
            normals.push([normal[0] / len, normal[1] / len, normal[2] / len]);
        } else {
            return Err(NormalEstimationError::DegenerateNeighborhood { index: i });
        }
    }

    Ok(PointCloud::new(
        points.clone(),
        cloud.colors().cloned(),
        Some(normals),
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    const EPS: f64 = 1e-6;

    #[test]
    fn test_estimate_normals_flat_plane() {
        let mut points = Vec::new();
        for x in -1..=1 {
            for y in -1..=1 {
                points.push([x as f64, y as f64, 0.0]);
            }
        }
        let cloud = PointCloud::new(points, None, None);

        let result = estimate_normals(&cloud, 5).unwrap();
        let normals = result.normals().unwrap();

        for n in normals {
            assert!((n[2].abs() - 1.0).abs() < EPS, "Normal z = {}", n[2]);
        }
    }

    #[test]
    fn test_estimate_normals_tilted_plane() {
        let mut points = Vec::new();
        for x in -1..=1 {
            for y in -1..=1 {
                points.push([x as f64, y as f64, (x + y) as f64]);
            }
        }
        let cloud = PointCloud::new(points, None, None);

        let result = estimate_normals(&cloud, 5).unwrap();
        let normals = result.normals().unwrap();

        let expected = [
            -1.0 / 3.0_f64.sqrt(),
            -1.0 / 3.0_f64.sqrt(),
            1.0 / 3.0_f64.sqrt(),
        ];
        for n in normals {
            let dot = n[0] * expected[0] + n[1] * expected[1] + n[2] * expected[2];
            assert!((dot.abs() - 1.0).abs() < EPS, "Normal dot = {}", dot);
        }
    }

    #[test]
    fn test_estimate_normals_edge_cases() {
        let cloud = PointCloud::new(Vec::new(), None, None);
        let result = estimate_normals(&cloud, 2);
        assert!(matches!(
            result,
            Err(NormalEstimationError::EmptyPointCloud)
        ));

        let points = vec![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]];
        let cloud = PointCloud::new(points, None, None);
        let result = estimate_normals(&cloud, 1);
        assert!(matches!(
            result,
            Err(NormalEstimationError::InvalidNeighborCount { .. })
        ));

        let result = estimate_normals(&cloud, 10);
        assert!(matches!(
            result,
            Err(NormalEstimationError::InvalidNeighborCount { .. })
        ));
    }
}
