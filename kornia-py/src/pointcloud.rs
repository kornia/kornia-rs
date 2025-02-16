use numpy::{PyArray2, PyArrayMethods};
use pyo3::prelude::*;

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
