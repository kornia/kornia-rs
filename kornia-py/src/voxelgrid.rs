use numpy::{PyArray2, PyArrayMethods};
use pyo3::prelude::*;
use pyo3::wrap_pyfunction;
use kornia_3d::voxelgrid::VoxelGrid;
use kornia_3d::pointcloud::PointCloud;

#[pyclass]
pub struct PyVoxelGrid {
    voxel_grid: VoxelGrid,
}

#[pymethods]
impl PyVoxelGrid {
    #[new]
    pub fn new(
        leaf_size: (f64, f64, f64),
        min_bounds: (f64, f64, f64),
        max_bounds: (f64, f64, f64),
    ) -> Self {
        Self {
            voxel_grid: VoxelGrid::new(
                [leaf_size.0, leaf_size.1, leaf_size.2],
                [min_bounds.0, min_bounds.1, min_bounds.2],
                [max_bounds.0, max_bounds.1, max_bounds.2],
            ),
        }
    }

    pub fn add_points(&mut self, py: Python, pointcloud: Py<PyArray2<f64>>) -> PyResult<()> {
        let rust_pointcloud = PointCloud::from_pypointcloud(pointcloud)?;
        self.voxel_grid.add_points(&rust_pointcloud);
        Ok(())
    }

    pub fn downsample(&self, py: Python) -> PyResult<Py<PyArray2<f64>>> {
        let downsampled = self.voxel_grid.downsample();
        let downsampled_points = downsampled
            .points()
            .iter()
            .flat_map(|p| p.iter().cloned())
            .collect::<Vec<f64>>();

        let array = PyArray2::from_vec(py, downsampled_points)
            .reshape((downsampled.len(), 3))
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyException, _>(format!("{}", e)))?;

        Ok(array.into_py(py))
    }
}

#[pymodule]
fn voxelgrid(py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<PyVoxelGrid>()?;
    Ok(())
}
