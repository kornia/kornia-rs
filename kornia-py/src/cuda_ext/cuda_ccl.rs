//! Device-resident GPU connected-component labeling.

use super::*;
use kornia_imgproc::connected_components::{connected_components as ccl_rs, Connectivity};

pub(crate) fn connected_components(
    py: Python<'_>,
    img: &PyImageApi,
    conn: Connectivity,
) -> PyResult<(i32, Py<PyAny>)> {
    let dev = img.as_device().ok_or_else(|| {
        PyValueError::new_err(
            "connected_components: expected a device Image; for a host image pass its numpy array",
        )
    })?;
    let Inner::U8C1(src) = dev else {
        return Err(PyValueError::new_err(format!(
            "connected_components: the GPU path supports u8 single-channel device images, \
             got {:?} with {} channel(s)",
            dev.dtype_enum(),
            dev.channels(),
        )));
    };
    let stream = source_stream(src)?;
    let mut labels = Image::<i32, 1>::zeros_cuda(src.size(), &stream).map_err(err)?;
    let n = ccl_rs(src, &mut labels, conn).map_err(err)?;
    let out = PyImageApi::from_device(Inner::I32C1(labels), img.color_space, device_mode::<i32>(1));
    Ok((n, Py::new(py, out)?.into_any()))
}
