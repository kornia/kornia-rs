use numpy::{PyArray2, PyArrayMethods};
use pyo3::prelude::*;
use std::collections::HashMap;

use kornia_3d::pointcloud::PointCloud;

pub type PyPointCloud = Py<PyArray2<f64>>;

pub trait FromPyPointCloud {
    fn from_pypointcloud(
        pointcloud: PyPointCloud,
    ) -> Result<PointCloud, Box<dyn std::error::Error>>;
}

impl FromPyPointCloud for PointCloud {
    fn from_pypointcloud(
        pointcloud: PyPointCloud,
    ) -> Result<PointCloud, Box<dyn std::error::Error>> {
        Python::with_gil(|py| {
            let array = pointcloud.bind(py);
            let data_slice = unsafe { array.as_slice() }
                .map_err(|e| PyErr::new::<pyo3::exceptions::PyException, _>(format!("{}", e)))?;

            let points = data_slice
                .chunks_exact(3)
                .map(|c| [c[0], c[1], c[2]])
                .collect();
            let pointcloud = PointCloud::new(points, None, None);
            Ok(pointcloud)
        })
    }
}

impl PointCloud {
    /// Applies a voxel grid filter to downsample the point cloud.
    ///
    /// # Arguments
    /// * `voxel_size` - The size of each voxel in the grid.
    ///
    /// # Returns
    /// A new `PointCloud` containing the downsampled points.
    /// Each point is assigned to a voxel based on its coordinates 
    /// divided by the voxel size
    pub fn voxel_grid_filter(&self, voxel_size: f64) -> PointCloud {
        // HashMap to store voxel indices and their corresponding points
        let mut voxel_map: HashMap<(i64, i64, i64), Vec<[f64; 3]>> = HashMap::new();

        // Iterate through all points in the point cloud
        for point in &self.points {
            // Compute the voxel index for the current point
            let voxel_index = (
                (point[0] / voxel_size).floor() as i64,
                (point[1] / voxel_size).floor() as i64,
                (point[2] / voxel_size).floor() as i64,
            );

            // Add the point to the corresponding voxel
            voxel_map
                .entry(voxel_index)
                .or_insert_with(Vec::new)
                .push(*point);
        }

        // Compute the representative point for each voxel (e.g., centroid)
        let downsampled_points: Vec<[f64; 3]> = voxel_map
            .values()
            .map(|points| {
                let mut centroid = [0.0, 0.0, 0.0];
                for point in points {
                    centroid[0] += point[0];
                    centroid[1] += point[1];
                    centroid[2] += point[2];
                }
                let n = points.len() as f64;
                centroid[0] /= n;
                centroid[1] /= n;
                centroid[2] /= n;
                centroid
            })
            .collect();

        // Return a new PointCloud with the downsampled points
        PointCloud::new(downsampled_points, None, None)
    }
}

#[pymethods]
impl PointCloud {
    #[staticmethod]
    pub fn voxel_grid_filter_py(
        py: Python,
        pointcloud: PyPointCloud,
        voxel_size: f64,
    ) -> PyResult<PyPointCloud> {
        let rust_pointcloud = PointCloud::from_pypointcloud(pointcloud)?;
        let downsampled = rust_pointcloud.voxel_grid_filter(voxel_size);
        // Convert back to PyPointCloud and return
        // (You would need to implement the reverse conversion)
        Ok(downsampled.to_pypointcloud(py))
    }
}
