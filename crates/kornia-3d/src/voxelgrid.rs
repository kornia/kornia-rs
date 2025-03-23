/// A voxel grid for organizing and downsampling point clouds.
pub struct VoxelGrid {
    leaf_size: [f64; 3],
    min_bounds: [f64; 3],
    max_bounds: [f64; 3],
    grid: std::collections::HashMap<[i64; 3], Vec<[f64; 3]>>,
}

impl VoxelGrid {
    /// Create a new VoxelGrid with the specified leaf size and bounds.
    pub fn new(leaf_size: [f64; 3], min_bounds: [f64; 3], max_bounds: [f64; 3]) -> Self {
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
            if self.is_within_bounds(point) {
                let voxel_index = self.compute_voxel_index(point);
                self.grid
                    .entry(voxel_index)
                    .or_insert_with(Vec::new)
                    .push(*point);
            }
        }
    }

    /// Compute the voxel index for a given point.
    fn compute_voxel_index(&self, point: &[f64; 3]) -> [i64; 3] {
        [
            ((point[0] - self.min_bounds[0]) / self.leaf_size[0]).floor() as i64,
            ((point[1] - self.min_bounds[1]) / self.leaf_size[1]).floor() as i64,
            ((point[2] - self.min_bounds[2]) / self.leaf_size[2]).floor() as i64,
        ]
    }

    /// Check if a point is within the bounds of the voxel grid.
    fn is_within_bounds(&self, point: &[f64; 3]) -> bool {
        (0..3).all(|i| point[i] >= self.min_bounds[i] && point[i] <= self.max_bounds[i])
    }

    /// Downsample the point cloud by averaging points in each voxel.
    pub fn downsample(&self) -> PointCloud {
        let mut downsampled_points = Vec::new();

        for points in self.grid.values() {
            let centroid = points.iter().fold([0.0; 3], |mut acc, p| {
                acc[0] += p[0];
                acc[1] += p[1];
                acc[2] += p[2];
                acc
            });
            let num_points = points.len() as f64;
            downsampled_points.push([
                centroid[0] / num_points,
                centroid[1] / num_points,
                centroid[2] / num_points,
            ]);
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

    let mut voxel_grid = VoxelGrid::new([1.0, 1.0, 1.0], [0.0, 0.0, 0.0], [2.0, 2.0, 2.0]);
    voxel_grid.add_points(&pointcloud);
    let downsampled = voxel_grid.downsample();

    assert_eq!(downsampled.len(), 2);
    assert!(downsampled.points().contains(&[0.05, 0.05, 0.05]));
    assert!(downsampled.points().contains(&[1.05, 1.05, 1.05]));
}