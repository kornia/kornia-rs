use numpy::{PyArray2, PyUntypedArrayMethods};
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
        Python::attach(|py| {
            let array = pointcloud.bind(py);
            if array.shape()[1] != 3 {
                return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                    "point cloud must be an (N, 3) float64 array",
                )
                .into());
            }
            // C-contiguous + aligned (an F-ordered array would be misread as rows).
            let data_slice = crate::pyutils::c_slice(array, "point cloud")?;

            let points = data_slice
                .chunks_exact(3)
                .map(|c| [c[0], c[1], c[2]])
                .collect();
            let pointcloud = PointCloud::new(points, None, None);
            Ok(pointcloud)
        })
    }
}
