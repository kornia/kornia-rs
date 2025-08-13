use std::collections::HashMap;
use glam::DVec3;
use crate::pointcloud::PointCloud;

/// Error returned when the leaf size is non-positive in any dimension.
#[derive(Debug, Clone, Copy)]
pub struct InvalidLeafSizeError;

impl std::fmt::Display for InvalidLeafSizeError {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(f, "Leaf size must be positive in all dimensions")
    }
}

impl std::error::Error for InvalidLeafSizeError {}

/// Data structure for voxel data when downsampling all data.
#[derive(Debug, Clone)]
struct VoxelData {
    /// Sum of point coordinates in the voxel.
    point_sum: DVec3,
    /// List of colors in the voxel.
    colors: Vec<[u8; 3]>,
    /// List of normals in the voxel.
    normals: Vec<[f64; 3]>,
    /// Number of points in the voxel.
    count: usize,
}

/// A 3D voxel grid for downsampling point clouds.
#[derive(Debug, Clone)]
pub struct VoxelGrid {
    /// The size of the voxel (leaf) in x, y, z dimensions.
    leaf_size: DVec3,
    /// Minimum number of points required per voxel.
    min_points_per_voxel: usize,
    /// Whether to downsample all data (points, colors, normals) or just points.
    downsample_all_data: bool,
}

impl VoxelGrid {
    /// Creates a new `VoxelGrid` with the specified leaf size.
    ///
    /// # Arguments
    /// * `leaf_size` - The size of the voxel in x, y, z dimensions.
    ///
    /// # Returns
    /// A `Result` containing the new `VoxelGrid` or an `InvalidLeafSizeError` if any dimension of `leaf_size` is non-positive.
    pub fn new(leaf_size: DVec3) -> Result<Self, InvalidLeafSizeError> {
        if leaf_size.cmple(DVec3::ZERO).any() {
            return Err(InvalidLeafSizeError);
        }
        Ok(Self {
            leaf_size,
            min_points_per_voxel: 1,
            downsample_all_data: true,
        })
    }

    /// Downsamples the input point cloud by grouping points into voxels and computing centroids.
    ///
    /// # Arguments
    /// * `point_cloud` - The input point cloud to downsample.
    ///
    /// # Returns
    /// A new `PointCloud` containing the downsampled points, and optionally colors and normals.
    pub fn downsample(&mut self, point_cloud: &PointCloud) -> PointCloud {
        if self.downsample_all_data {
            let mut grid: HashMap<(i32, i32, i32), VoxelData> = HashMap::new();

            // Group points, colors, and normals into voxels
            for (i, point) in point_cloud.points().iter().enumerate() {
                let point_vec = DVec3::from_array(*point);
                let idx_vec = (point_vec / self.leaf_size).floor();
                let key = (idx_vec.x as i32, idx_vec.y as i32, idx_vec.z as i32);
                let entry = grid.entry(key).or_insert(VoxelData {
                    point_sum: DVec3::ZERO,
                    colors: Vec::new(),
                    normals: Vec::new(),
                    count: 0,
                });
                entry.point_sum += point_vec;
                entry.count += 1;

                if let Some(colors) = point_cloud.colors() {
                    if let Some(color) = colors.get(i) {
                        entry.colors.push(*color);
                    }
                }
                if let Some(normals) = point_cloud.normals() {
                    if let Some(normal) = normals.get(i) {
                        entry.normals.push(*normal);
                    }
                }
            }

            // Compute centroids and normalized colors/normals for each voxel
            let mut points = Vec::new();
            let mut colors = if point_cloud.colors().is_some() {
                Some(Vec::new())
            } else {
                None
            };
            let mut normals = if point_cloud.normals().is_some() {
                Some(Vec::new())
            } else {
                None
            };

            for (_key, voxel_data) in grid {
                if voxel_data.count >= self.min_points_per_voxel {
                    let inv_count = 1.0 / voxel_data.count as f64;
                    let centroid = voxel_data.point_sum * inv_count;
                    points.push([centroid.x, centroid.y, centroid.z]);

                    if let Some(ref mut colors_vec) = colors {
                        if !voxel_data.colors.is_empty() {
                            let color_sum: [u64; 3] = voxel_data.colors.iter().fold(
                                [0, 0, 0],
                                |acc, c| [acc[0] + c[0] as u64, acc[1] + c[1] as u64, acc[2] + c[2] as u64],
                            );
                            colors_vec.push([
                                (color_sum[0] as f64 * inv_count).round().clamp(0.0, 255.0) as u8,
                                (color_sum[1] as f64 * inv_count).round().clamp(0.0, 255.0) as u8,
                                (color_sum[2] as f64 * inv_count).round().clamp(0.0, 255.0) as u8,
                            ]);
                        }
                    }
                    if let Some(ref mut normals_vec) = normals {
                        if !voxel_data.normals.is_empty() {
                            let normal_sum: [f64; 3] = voxel_data.normals.iter().fold(
                                [0.0, 0.0, 0.0],
                                |acc, n| [acc[0] + n[0], acc[1] + n[1], acc[2] + n[2]],
                            );
                            let normal = [
                                normal_sum[0] * inv_count,
                                normal_sum[1] * inv_count,
                                normal_sum[2] * inv_count,
                            ];
                            let norm = (normal[0] * normal[0] + normal[1] * normal[1] + normal[2] * normal[2]).sqrt();
                            if norm > 0.0 {
                                normals_vec.push([normal[0] / norm, normal[1] / norm, normal[2] / norm]);
                            } else {
                                normals_vec.push(normal);
                            }
                        }
                    }
                }
            }

            PointCloud::new(points, colors, normals)
        } else {
            let mut grid: HashMap<(i32, i32, i32), (DVec3, usize)> = HashMap::new();

            // Group points into voxels
            for point in point_cloud.points().iter() {
                let point_vec = DVec3::from_array(*point);
                let idx_vec = (point_vec / self.leaf_size).floor();
                let key = (idx_vec.x as i32, idx_vec.y as i32, idx_vec.z as i32);
                let entry = grid.entry(key).or_insert((DVec3::ZERO, 0));
                entry.0 += point_vec;
                entry.1 += 1;
            }

            // Compute centroids for each voxel
            let mut points = Vec::new();
            for (_key, (sum, count)) in grid {
                if count >= self.min_points_per_voxel {
                    let inv_count = 1.0 / count as f64;
                    let centroid = sum * inv_count;
                    points.push([centroid.x, centroid.y, centroid.z]);
                }
            }

            PointCloud::new(points, None, None)
        }
    }

    /// Gets the minimum bound of the point cloud.
    ///
    /// # Arguments
    /// * `point_cloud` - The input point cloud.
    ///
    /// # Returns
    /// The minimum x, y, z coordinates as a `DVec3`. Returns `[0.0, 0.0, 0.0]` if empty.
    #[inline]
    pub fn get_min_bound(&self, point_cloud: &PointCloud) -> DVec3 {
        if point_cloud.points().is_empty() {
            DVec3::ZERO
        } else {
            point_cloud
                .points()
                .iter()
                .map(|&p| DVec3::from_array(p))
                .fold(DVec3::splat(f64::INFINITY), |a, b| a.min(b))
        }
    }

    /// Gets the maximum bound of the point cloud.
    ///
    /// # Arguments
    /// * `point_cloud` - The input point cloud.
    ///
    /// # Returns
    /// The maximum x, y, z coordinates as a `DVec3`. Returns `[0.0, 0.0, 0.0]` if empty.
    #[inline]
    pub fn get_max_bound(&self, point_cloud: &PointCloud) -> DVec3 {
        if point_cloud.points().is_empty() {
            DVec3::ZERO
        } else {
            point_cloud
                .points()
                .iter()
                .map(|&p| DVec3::from_array(p))
                .fold(DVec3::splat(f64::NEG_INFINITY), |a, b| a.max(b))
        }
    }

    /// Computes the voxel index for a given point.
    ///
    /// # Arguments
    /// * `point` - The input point as `[x, y, z]`.
    ///
    /// # Returns
    /// The voxel index as `(i32, i32, i32)`.
    #[inline]
    pub fn get_voxel_index(&self, point: &[f64; 3]) -> (i32, i32, i32) {
        let point_vec = DVec3::from_array(*point);
        let idx_vec = (point_vec / self.leaf_size).floor();
        (idx_vec.x as i32, idx_vec.y as i32, idx_vec.z as i32)
    }

    /// Sets the voxel grid leaf size.
    ///
    /// # Arguments
    /// * `leaf_size` - The new leaf size in x, y, z dimensions.
    ///
    /// # Returns
    /// A `Result` indicating success or an `InvalidLeafSizeError` if any dimension of `leaf_size` is non-positive.
    pub fn set_leaf_size(&mut self, leaf_size: DVec3) -> Result<(), InvalidLeafSizeError> {
        if leaf_size.cmple(DVec3::ZERO).any() {
            return Err(InvalidLeafSizeError);
        }
        self.leaf_size = leaf_size;
        Ok(())
    }

    /// Gets the voxel grid leaf size.
    ///
    /// # Returns
    /// The current leaf size as a `DVec3`.
    #[inline]
    pub fn get_leaf_size(&self) -> DVec3 {
        self.leaf_size
    }

    /// Sets the minimum number of points required per voxel.
    ///
    /// # Arguments
    /// * `min_points` - The minimum number of points.
    #[inline]
    pub fn set_min_points_per_voxel(&mut self, min_points: usize) {
        self.min_points_per_voxel = min_points;
    }

    /// Gets the minimum number of points required per voxel.
    ///
    /// # Returns
    /// The current minimum number of points.
    #[inline]
    pub fn get_min_points_per_voxel(&self) -> usize {
        self.min_points_per_voxel
    }

    /// Sets whether to downsample all data (points, colors, normals) or just points.
    ///
    /// # Arguments
    /// * `downsample` - If `true`, downsample all data; if `false`, only points.
    #[inline]
    pub fn set_downsample_all_data(&mut self, downsample: bool) {
        self.downsample_all_data = downsample;
    }

    /// Gets whether all data (points, colors, normals) is downsampled.
    ///
    /// # Returns
    /// `true` if all data is downsampled, `false` if only points.
    #[inline]
    pub fn get_downsample_all_data(&self) -> bool {
        self.downsample_all_data
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use glam::DVec3;

    #[test]
    fn test_new_voxel_grid() {
        let leaf_size = DVec3::from_array([1.0, 1.0, 1.0]);
        let voxel_grid = VoxelGrid::new(leaf_size).unwrap();
        assert_eq!(voxel_grid.leaf_size.x, leaf_size.x);
        assert_eq!(voxel_grid.leaf_size.y, leaf_size.y);
        assert_eq!(voxel_grid.leaf_size.z, leaf_size.z);
    }

    #[test]
    fn test_new_invalid_leaf_size() {
        let leaf_size = DVec3::from_array([0.0, 1.0, 1.0]);
        assert!(VoxelGrid::new(leaf_size).is_err());
    }

    #[test]
    fn test_downsample_points_only() {
        let points = vec![[1.0, 1.0, 1.0], [1.1, 1.1, 1.1]];
        let colors = Some(vec![[255, 0, 0], [0, 255, 0]]);
        let normals = Some(vec![[0.0, 1.0, 0.0], [0.0, 1.0, 0.0]]);
        let point_cloud = PointCloud::new(points, colors, normals);
        let leaf_size = DVec3::from_array([1.0, 1.0, 1.0]);
        let mut voxel_grid = VoxelGrid::new(leaf_size).unwrap();
        voxel_grid.set_downsample_all_data(false);

        let downsampled = voxel_grid.downsample(&point_cloud);
        assert_eq!(downsampled.len(), 1);
        let centroid = downsampled.points()[0];
        assert!((centroid[0] - 1.05).abs() < 0.01);
        assert!((centroid[1] - 1.05).abs() < 0.01);
        assert!((centroid[2] - 1.05).abs() < 0.01);
        assert!(downsampled.colors().is_none());
        assert!(downsampled.normals().is_none());
    }

    #[test]
    fn test_downsample_with_colors_and_normals() {
        let points = vec![[1.0, 1.0, 1.0], [1.1, 1.1, 1.1]];
        let colors = Some(vec![[255, 0, 0], [0, 255, 0]]);
        let normals = Some(vec![[0.0, 1.0, 0.0], [0.0, 1.0, 0.0]]);
        let point_cloud = PointCloud::new(points, colors, normals);
        let leaf_size = DVec3::from_array([1.0, 1.0, 1.0]);
        let mut voxel_grid = VoxelGrid::new(leaf_size).unwrap();
        voxel_grid.set_downsample_all_data(true);

        let downsampled = voxel_grid.downsample(&point_cloud);
        assert_eq!(downsampled.len(), 1);
        let centroid = downsampled.points()[0];
        assert!((centroid[0] - 1.05).abs() < 0.01);
        assert!((centroid[1] - 1.05).abs() < 0.01);
        assert!((centroid[2] - 1.05).abs() < 0.01);

        if let Some(colors) = downsampled.colors() {
            assert_eq!(colors.len(), 1);
            assert_eq!(colors[0], [128, 128, 0]);
        }
        if let Some(normals) = downsampled.normals() {
            assert_eq!(normals.len(), 1);
            assert!((normals[0][1] - 1.0).abs() < 0.01);
            let norm = (normals[0][0] * normals[0][0] + normals[0][1] * normals[0][1] + normals[0][2] * normals[0][2]).sqrt();
            assert!((norm - 1.0).abs() < 0.01); // Verify normal is unit length
        }
    }

    #[test]
    fn test_bounds() {
        let point_cloud = PointCloud::new(vec![[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]], None, None);
        let voxel_grid = VoxelGrid::new(DVec3::from_array([1.0, 1.0, 1.0])).unwrap();
        assert_eq!(
            voxel_grid.get_min_bound(&point_cloud),
            DVec3::from_array([0.0, 0.0, 0.0])
        );
        assert_eq!(
            voxel_grid.get_max_bound(&point_cloud),
            DVec3::from_array([1.0, 1.0, 1.0])
        );
    }

    #[test]
    fn test_min_points_per_voxel() {
        let points = vec![[1.0, 1.0, 1.0]];
        let point_cloud = PointCloud::new(points, None, None);
        let leaf_size = DVec3::from_array([1.0, 1.0, 1.0]);
        let mut voxel_grid = VoxelGrid::new(leaf_size).unwrap();
        voxel_grid.set_min_points_per_voxel(2);
        assert_eq!(voxel_grid.get_min_points_per_voxel(), 2);

        let downsampled = voxel_grid.downsample(&point_cloud);
        assert_eq!(downsampled.len(), 0);
    }

    #[test]
    fn test_set_get_leaf_size() {
        let mut voxel_grid = VoxelGrid::new(DVec3::from_array([1.0, 1.0, 1.0])).unwrap();
        let new_leaf_size = DVec3::from_array([2.0, 2.0, 2.0]);
        voxel_grid.set_leaf_size(new_leaf_size).unwrap();
        let retrieved = voxel_grid.get_leaf_size();
        assert_eq!(retrieved.x, new_leaf_size.x);
        assert_eq!(retrieved.y, new_leaf_size.y);
        assert_eq!(retrieved.z, new_leaf_size.z);
    }

    #[test]
    fn test_set_invalid_leaf_size() {
        let mut voxel_grid = VoxelGrid::new(DVec3::from_array([1.0, 1.0, 1.0])).unwrap();
        assert!(voxel_grid.set_leaf_size(DVec3::from_array([-1.0, 1.0, 1.0])).is_err());
    }

    #[test]
    fn test_set_get_downsample_all_data() {
        let mut voxel_grid = VoxelGrid::new(DVec3::from_array([1.0, 1.0, 1.0])).unwrap();
        voxel_grid.set_downsample_all_data(false);
        assert!(!voxel_grid.get_downsample_all_data());
        voxel_grid.set_downsample_all_data(true);
        assert!(voxel_grid.get_downsample_all_data());
    }
}