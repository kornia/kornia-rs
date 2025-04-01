use glam::Vec3;

/// A voxel grid for organizing and downsampling point clouds.
pub struct VoxelGrid {
    leaf_size: Vec3,
    min_bounds: Vec3,
    max_bounds: Vec3,
    grid: std::collections::HashMap<[i64; 3], Vec<Vec3>>,
}

impl VoxelGrid {
    /// Create a new VoxelGrid with the specified leaf size and bounds.
    pub fn new(leaf_size: Vec3, min_bounds: Vec3, max_bounds: Vec3) -> Self {
        Self {
            leaf_size,
            min_bounds,
            max_bounds,
            grid: std::collections::HashMap::new(),
        }
    }

    /// Add points from a point cloud to the voxel grid.
    pub fn add_points(&mut self, pointcloud: &PointCloud) {
        for point in pointcloud.points() {
            let point_vec = Vec3::from(*point);
            if self.is_within_bounds(&point_vec) {
                let voxel_index = self.compute_voxel_index(&point_vec);
                self.grid
                    .entry(voxel_index)
                    .or_insert_with(Vec::new)
                    .push(point_vec);
            }
        }
    }

    /// Compute the voxel index for a given point.
    fn compute_voxel_index(&self, point: &Vec3) -> [i64; 3] {
        let index = (*point - self.min_bounds) / self.leaf_size;
        [index.x.floor() as i64, index.y.floor() as i64, index.z.floor() as i64]
    }

    /// Check if a point is within the bounds of the voxel grid.
    fn is_within_bounds(&self, point: &Vec3) -> bool {
        (0..3).all(|i| point[i] >= self.min_bounds[i] && point[i] <= self.max_bounds[i])
    }

    /// Downsample the point cloud by averaging points in each voxel.
    pub fn downsample(&self) -> PointCloud {
        let mut downsampled_points = Vec::new();

        for points in self.grid.values() {
            let centroid = points.iter().fold(Vec3::ZERO, |acc, p| acc + *p);
            let num_points = points.len() as f32;
            downsampled_points.push((centroid / num_points).into());
        }

        PointCloud::new(downsampled_points, None, None)
    }
}


#[test]
fn test_voxel_grid() {
    let pointcloud = PointCloud::new(
        vec![
            [0.0, 0.0, 0.0],
            [0.1, 0.1, 0.1],
            [1.0, 1.0, 1.0],
            [1.1, 1.1, 1.1],
        ],
        None,
        None,
    );

    let mut voxel_grid = VoxelGrid::new(Vec3::ONE, Vec3::ZERO, Vec3::ONE * 2.0);
    voxel_grid.add_points(&pointcloud);
    let downsampled = voxel_grid.downsample();

    assert_eq!(downsampled.len(), 2);
    assert!(downsampled.points().contains(&[0.05, 0.05, 0.05]));
    assert!(downsampled.points().contains(&[1.05, 1.05, 1.05]));
}