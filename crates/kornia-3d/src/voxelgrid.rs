use glam::Vec3;
use std::collections::HashMap;

/// A voxel grid for organizing and downsampling point clouds.
pub struct VoxelGrid {
    voxel_size: f32,
    origin: Vec3,
    voxels: HashMap<[i64; 3], Voxel>,
}

#[derive(Clone)]
pub struct Voxel {
    pub grid_index: [i64; 3],
    pub center: Vec3,
}

impl VoxelGrid {
    /// Create a new VoxelGrid with the specified voxel size and origin.
    pub fn new(voxel_size: f32, origin: Vec3) -> Self {
        Self {
            voxel_size,
            origin,
            voxels: HashMap::new(),
        }
    }

    /// Clear all voxels in the grid.
    pub fn clear(&mut self) {
        self.voxels.clear();
    }

    /// Check if the voxel grid is empty.
    pub fn is_empty(&self) -> bool {
        self.voxels.is_empty()
    }

    /// Get the minimum bound of the voxel grid.
    pub fn get_min_bound(&self) -> Vec3 {
        if self.voxels.is_empty() {
            self.origin
        } else {
            let mut min_grid_index = *self.voxels.keys().next().unwrap();
            for &grid_index in self.voxels.keys() {
                min_grid_index = [
                    min_grid_index[0].min(grid_index[0]),
                    min_grid_index[1].min(grid_index[1]),
                    min_grid_index[2].min(grid_index[2]),
                ];
            }
            Vec3::new(
                min_grid_index[0] as f32,
                min_grid_index[1] as f32,
                min_grid_index[2] as f32,
            ) * self.voxel_size
                + self.origin
        }
    }

    /// Get the maximum bound of the voxel grid.
    pub fn get_max_bound(&self) -> Vec3 {
        if self.voxels.is_empty() {
            self.origin
        } else {
            let mut max_grid_index = *self.voxels.keys().next().unwrap();
            for &grid_index in self.voxels.keys() {
                max_grid_index = [
                    max_grid_index[0].max(grid_index[0]),
                    max_grid_index[1].max(grid_index[1]),
                    max_grid_index[2].max(grid_index[2]),
                ];
            }
            (Vec3::new(
                max_grid_index[0] as f32,
                max_grid_index[1] as f32,
                max_grid_index[2] as f32,
            ) + Vec3::splat(1.0))
                * self.voxel_size
                + self.origin
        }
    }

    /// Add a voxel with the specified grid index.
    pub fn add_voxel(&mut self, grid_index: [i64; 3]) {
        let center = self.get_voxel_center_coordinate(grid_index);
        self.voxels.insert(grid_index, Voxel { grid_index, center });
    }

    /// Remove a voxel with the specified grid index.
    pub fn remove_voxel(&mut self, grid_index: [i64; 3]) {
        self.voxels.remove(&grid_index);
    }

    /// Get the voxel index for a given point.
    pub fn get_voxel(&self, point: &Vec3) -> Option<[i64; 3]> {
        let relative_point = (*point - self.origin) / self.voxel_size;
        Some([
            relative_point.x.floor() as i64,
            relative_point.y.floor() as i64,
            relative_point.z.floor() as i64,
        ])
    }

    /// Get the center coordinate of a voxel given its index.
    pub fn get_voxel_center_coordinate(&self, grid_index: [i64; 3]) -> Vec3 {
        let grid_index_vec = Vec3::new(grid_index[0] as f32, grid_index[1] as f32, grid_index[2] as f32);
        (grid_index_vec + Vec3::splat(0.5)) * self.voxel_size + self.origin
    }

    /// Downsample the point cloud by averaging points in each voxel.
    pub fn downsample(&self, pointcloud: &PointCloud) -> PointCloud {
        let mut voxel_points: HashMap<[i64; 3], Vec<Vec3>> = HashMap::new();

        for point in pointcloud.points() {
            let point_vec = PointCloud::point_to_vec3(point);
            if let Some(voxel_index) = self.get_voxel(&point_vec) {
                voxel_points
                    .entry(voxel_index)
                    .or_insert_with(Vec::new)
                    .push(point_vec);
            }
        }

        let downsampled_points = voxel_points
            .values()
            .map(|points| {
                let centroid = points.iter().fold(Vec3::ZERO, |acc, p| acc + *p) / points.len() as f32;
                PointCloud::vec3_to_point(centroid)
            })
            .collect();

        PointCloud::new(downsampled_points, None, None)
    }

    /// Create a VoxelGrid from a point cloud within specified bounds.
    pub fn create_from_pointcloud_within_bounds(
        pointcloud: &PointCloud,
        voxel_size: f32,
        min_bound: Vec3,
        max_bound: Vec3,
    ) -> Self {
        if voxel_size <= 0.0 {
            panic!("voxel_size must be greater than 0.");
        }

        if voxel_size * (i64::MAX as f32) < (max_bound - min_bound).max_element() {
            panic!("voxel_size is too small.");
        }

        let mut voxel_grid = Self::new(voxel_size, min_bound);
        let mut voxel_points: HashMap<[i64; 3], Vec<Vec3>> = HashMap::new();

        for point in pointcloud.points() {
            let point_vec = PointCloud::point_to_vec3(point);
            if point_vec.cmpge(min_bound).all() && point_vec.cmple(max_bound).all() {
                let relative_point = (point_vec - min_bound) / voxel_size;
                let voxel_index = [
                    relative_point.x.floor() as i64,
                    relative_point.y.floor() as i64,
                    relative_point.z.floor() as i64,
                ];
                voxel_points
                    .entry(voxel_index)
                    .or_insert_with(Vec::new)
                    .push(point_vec);
            }
        }

        for (voxel_index, points) in voxel_points {
            let centroid = points.iter().fold(Vec3::ZERO, |acc, p| acc + *p) / points.len() as f32;
            voxel_grid.voxels.insert(
                voxel_index,
                Voxel {
                    grid_index: voxel_index,
                    center: centroid,
                },
            );
        }

        voxel_grid
    }

    /// Create a VoxelGrid from a point cloud with automatic bounds.
    pub fn create_from_pointcloud(
        pointcloud: &PointCloud,
        voxel_size: f32,
    ) -> Self {
        let min_bound = pointcloud.get_min_bound() - Vec3::splat(voxel_size * 0.5);
        let max_bound = pointcloud.get_max_bound() + Vec3::splat(voxel_size * 0.5);
        Self::create_from_pointcloud_within_bounds(pointcloud, voxel_size, min_bound, max_bound)
    }
}

#[test]
fn test_voxel_grid() {
    // Test empty VoxelGrid
    let empty_voxel_grid = VoxelGrid::new(1.0, Vec3::ZERO);
    assert!(empty_voxel_grid.is_empty());
    assert_eq!(empty_voxel_grid.get_min_bound(), Vec3::ZERO);
    assert_eq!(empty_voxel_grid.get_max_bound(), Vec3::ZERO);

    // Test adding and removing voxels
    let mut voxel_grid = VoxelGrid::new(1.0, Vec3::ZERO);
    voxel_grid.add_voxel([0, 0, 0]);
    voxel_grid.add_voxel([1, 1, 1]);

    assert_eq!(voxel_grid.is_empty(), false);
    assert_eq!(voxel_grid.get_min_bound(), Vec3::ZERO);
    assert_eq!(voxel_grid.get_max_bound(), Vec3::new(2.0, 2.0, 2.0));

    voxel_grid.remove_voxel([0, 0, 0]);
    assert_eq!(voxel_grid.get_min_bound(), Vec3::new(1.0, 1.0, 1.0));

    // Test downsampling
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

    let downsampled = voxel_grid.downsample(&pointcloud);
    assert_eq!(downsampled.len(), 1); // Only one voxel remains
    assert!(downsampled.points().contains(&[1.05, 1.05, 1.05]));

    // Test create_from_pointcloud
    let voxel_grid_from_pc = VoxelGrid::create_from_pointcloud(&pointcloud, 1.0);
    assert_eq!(voxel_grid_from_pc.is_empty(), false);
    assert_eq!(voxel_grid_from_pc.get_min_bound(), Vec3::ZERO);
    assert_eq!(voxel_grid_from_pc.get_max_bound(), Vec3::new(2.0, 2.0, 2.0));
}