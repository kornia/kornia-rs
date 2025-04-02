use numpy::{PyArray2, PyArrayMethods};
use pyo3::prelude::*;
use pyo3::wrap_pyfunction;
use kornia_3d::voxelgrid::VoxelGrid;
use kornia_3d::pointcloud::PointCloud;
use glam::Vec3;
use std::collections::HashMap;

#[pyclass]
pub struct PyVoxelGrid {
    voxel_grid: VoxelGrid,
}

#[pymethods]
impl PyVoxelGrid {
    #[new]
    pub fn new(
        voxel_size: f32,
        origin: (f32, f32, f32),
    ) -> Self {
        Self {
            voxel_grid: VoxelGrid::new(
                voxel_size,
                Vec3::new(origin.0, origin.1, origin.2),
            ),
        }
    }

    pub fn add_points(&mut self, py: Python, pointcloud: Py<PyArray2<f64>>) -> PyResult<()> {
        let rust_pointcloud = PointCloud::from_pypointcloud(pointcloud)?;
        self.voxel_grid = VoxelGrid::create_from_pointcloud(&rust_pointcloud, self.voxel_grid.voxel_size);
        Ok(())
    }

    pub fn downsample(&self, py: Python, pointcloud: Py<PyArray2<f64>>) -> PyResult<Py<PyArray2<f64>>> {
        let rust_pointcloud = PointCloud::from_pypointcloud(pointcloud)?;
        let downsampled = self.voxel_grid.downsample(&rust_pointcloud);
        let downsampled_points = downsampled
            .points()
            .iter()
            .flat_map(|p| p.iter().cloned())
            .collect::<Vec<f64>>();
            
        if downsampled_points.len() % 3 != 0 {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "Point cloud data is corrupted: number of coordinates is not divisible by 3"
            ));
        }

        let array = PyArray2::from_vec(py, downsampled_points)
            .reshape((downsampled.len(), 3))
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("{}", e)))?;

        Ok(array.into_py(py))
    }
}

#[pymodule]
fn voxelgrid(py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<PyVoxelGrid>()?;
    Ok(())
}
