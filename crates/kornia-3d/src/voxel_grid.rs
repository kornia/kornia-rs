use std::collections::HashMap;
use crate::{DVec3, PointCloud};


#[derive(Debug, Clone)]
pub struct VoxelGrid {
    // The size of the voxel (leaf) in x, y, z, dimensions
    leaf_size: DVec3,
    min_points_per_voxel: usize,
    downsample_all_data: bool,
}


impl VoxelGrid {
    pub fn new(leaf_size: DVec3) -> Self {
        if leaf_size.x <= 0.0 || leaf_size.y <= 0.0 || leaf_size.z <= 0.0 {
            panic!("Leaf size must be positive in all dimensions");
        }

        VoxelGrid { 
            leaf_size,
            min_points_per_voxel: 1,
            downsample_all_data: true,
        }
    }

    pub fn downsample(&mut self, point_cloud: &PointCloud) -> PointCloud {
        let mut grid: HashMap<(i32, i32, i32), (DVec3, Vec<[u8; 3]>, Vec<[f64; 3]>, usize)> =
            HashMap::new();

        // Group points into voxels
        for (i, point) in point_cloud.points().iter().enumerate() {
            let point_vec = DVec3::from_array(point);
            let ix = (point_vec.x / self.leaf_size.x).floor() as i32;
            let iy = (point_vec.y / self.leaf_size.y).floor() as i32;
            let iz = (point_vec.z / self.leaf_size.z).floor() as i32;
            let key = (ix, iy, iz);
            let entry = grid.entry(key).or_insert((DVec3::new(0.0, 0.0, 0.0), Vec::new(), Vec::new(), 0));
            entry.0.x += point_vec.x;
            entry.0.y += point_vec.y;
            entry.0.z += point_vec.z;
            entry.3 += 1;

            if self.downsample_all_data {
                if let Some(colors) = point_cloud.colors() {
                    if let Some(color) = colors.get(i) {
                        entry.1.push(*color);
                    }
                }
                if let Some(normals) = point_cloud.normals() {
                    if let Some(normal) = normals.get(i) {
                        entry.2.push(*normal);
                    }
                }
            }
        }

        // Compute centroids for each voxel
        let mut points = Vec::new();
        let mut colors = if self.downsample_all_data && point_cloud.colors().is_some() {
            Some(Vec::new())
        } else {
            None
        };
        let mut normals = if self.downsample_all_data && point_cloud.normals().is_some() {
            Some(Vec::new())
        } else {
            None
        };

        for (_key, (sum, color_vec, normal_vec, count)) in grid {
            if count >= self.min_points_per_voxel {
                let inv_count = 1.0 / count as f64;
                let centroid = DVec3::new(sum.x * inv_count, sum.y * inv_count, sum.z * inv_count);
                points.push([centroid.x, centroid.y, centroid.z]);

                if self.downsample_all_data {
                    if let Some(ref mut colors_vec) = colors {
                        if !color_vec.is_empty() {
                            let color_sum: [u64; 3] = color_vec.iter().fold(
                                [0, 0, 0],
                                |acc, c| [acc[0] + c[0] as u64, acc[1] + c[1] as u64, acc[2] + c[2] as u64],
                            );
                            colors_vec.push([
                                (color_sum[0] as f64 * inv_count).round() as u8,
                                (color_sum[1] as f64 * inv_count).round() as u8,
                                (color_sum[2] as f64 * inv_count).round() as u8,
                            ]);
                        }
                    }
                    if let Some(ref mut normals_vec) = normals {
                        if !normal_vec.is_empty() {
                            let normal_sum: [f64; 3] = normal_vec.iter().fold(
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
        }

        PointCloud::new(points, colors, normals)
    }

    
    // Set voxel grid leaf size
    pub fn set_leaf_size(&mut self, leaf_size: DVec3) {
        if leaf_size.x <= 0.0 || leaf_size.y <= 0.0 || leaf_size.z <= 0.0 {
            panic!("Leaf size must be positive in all dimensions");
        }
        self.leaf_size = leaf_size;
    }

    // Set voxel grid leaf size
    pub fn get_leaf_size(&mut self) -> DVec3 {
        self.leaf_size.clone()
    }

    // Set min points per voxel
    pub fn set_min_points_per_voxel(&mut self, min_points: usize) {
        self.min_points_per_voxel = min_points;
    }

    // Get min points per voxel
    pub fn get_min_points_per_voxel(&mut self) {
        self.min_points_per_voxel
    }

    // Set whether to downsample all data (points, colors, normals) or just points.
    pub fn set_downsample_all_data(&mut self, downsample: bool) {
        self.downsample_all_data = downsample;
    }

    // Get whether all data (points, colors, normals) is downsampled.
    pub fn get_downsample_all_data(&self) -> bool {
        self.downsample_all_data
    }

    
}



//-------------------------- Tests ----------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_new_voxel_grid() {
        let leaf_size = DVec3::new(1.0, 1.0, 1.0);
        let voxel_grid = VoxelGrid::new(leaf_size.clone());
        assert_eq!(voxel_grid.leaf_size.x, leaf_size.x);
        assert_eq!(voxel_grid.leaf_size.y, leaf_size.y);
        assert_eq!(voxel_grid.leaf_size.z, leaf_size.z);

    }

    #[test]
    fn test_downsample_points_only() {
        let points = vec![[1.0, 1.0, 1.0], [1.1, 1.1, 1.1]];
        let point_cloud = PointCloud::new(points, None, None);
        let leaf_size = DVec3::new(1.0, 1.0, 1.0);
        let mut voxel_grid = VoxelGrid::new(leaf_size);

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
        let leaf_size = DVec3::new(1.0, 1.0, 1.0);
        let mut voxel_grid = VoxelGrid::new(leaf_size);

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
        }
    }

    #[test]
    fn test_min_points_per_voxel() {
        let points = vec![[1.0, 1.0, 1.0]];
        let point_cloud = PointCloud::new(points, None, None);
        let leaf_size = DVec3::new(1.0, 1.0, 1.0);
        let mut voxel_grid = VoxelGrid::new(leaf_size);
        voxel_grid.set_minimum_points_per_voxel(2);

        let downsampled = voxel_grid.downsample(&point_cloud);
        assert_eq!(downsampled.len(), 0);
    }

    #[test]
    fn test_set_get_leaf_size() {
        let mut voxel_grid = VoxelGrid::new(DVec3::new(1.0, 1.0, 1.0));
        let new_leaf_size = DVec3::new(2.0, 2.0, 2.0);
        voxel_grid.set_leaf_size(new_leaf_size.clone());
        let retrieved = voxel_grid.get_leaf_size();
        assert_eq!(retrieved.x, new_leaf_size.x);
        assert_eq!(retrieved.y, new_leaf_size.y);
        assert_eq!(retrieved.z, new_leaf_size.z);
    }

    #[test]
    fn test_set_get_downsample_all_data() {
        let mut voxel_grid = VoxelGrid::new(DVec3::new(1.0, 1.0, 1.0));
        voxel_grid.set_downsample_all_data(false);
        assert_eq!(voxel_grid.get_downsample_all_data(), false);
        voxel_grid.set_downsample_all_data(true);
        assert_eq!(voxel_grid.get_downsample_all_data(), true);
    }

    #[test]
    #[should_panic]
    fn test_invalid_leaf_size() {
        let leaf_size = DVec3::new(0.0, 1.0, 1.0);
        VoxelGrid::new(leaf_size);
    }

    #[test]
    #[should_panic]
    fn test_set_invalid_leaf_size() {
        let mut voxel_grid = VoxelGrid::new(DVec3::new(1.0, 1.0, 1.0));
        voxel_grid.set_leaf_size(DVec3::new(-1.0, 1.0, 1.0));
    }
}