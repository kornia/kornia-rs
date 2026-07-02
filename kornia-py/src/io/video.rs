use crate::image::PyImageApi;
use kornia_io::stream::video::{ImageFormat, VideoReader};
use numpy::{PyArray, PyArrayMethods};
use numpy::{PyArray, PyArrayMethods};
use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;
use std::sync::{Mutex, MutexGuard};

#[pyclass(name = "ImageFormat")]
#[derive(Clone, Copy)]
pub enum PyImageFormat {
    Rgb8,
    Mono8,
}

impl From<PyImageFormat> for ImageFormat {
    fn from(value: PyImageFormat) -> Self {
        match value {
            PyImageFormat::Rgb8 => ImageFormat::Rgb8,
            PyImageFormat::Mono8 => ImageFormat::Mono8,
        }
    }
}

#[pyclass(name = "VideoReader")]
pub struct PyVideoReader {
    reader: Mutex<VideoReader>,
    format: PyImageFormat,
}

impl PyVideoReader {
    fn lock_reader(&self) -> PyResult<MutexGuard<'_, VideoReader>> {
        self.reader
            .lock()
            .map_err(|_| PyRuntimeError::new_err("VideoReader mutex was poisoned"))
    }
}

#[pymethods]
impl PyVideoReader {
    #[new]
    pub fn new(path: &str, format: PyImageFormat) -> PyResult<Self> {
        if matches!(format, PyImageFormat::Mono8) {
            return Err(PyRuntimeError::new_err(
                "Mono8 video reading is not supported yet",
            ));
        }

        let reader = VideoReader::new(path, format.into())
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;

        Ok(Self {
            reader: Mutex::new(reader),
            format,
        })
    }

    pub fn start(&self) -> PyResult<()> {
        self.lock_reader()?
            .start()
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))
    }

    pub fn pause(&self) -> PyResult<()> {
        self.lock_reader()?
            .pause()
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))
    }

    pub fn close(&self) -> PyResult<()> {
        self.lock_reader()?
            .close()
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))
    }

    pub fn reset(&self) -> PyResult<()> {
        self.lock_reader()?
            .reset()
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))
    }

    #[getter]
    pub fn fps(&self) -> PyResult<Option<f64>> {
        Ok(self.lock_reader()?.get_fps())
    }

    #[getter]
    pub fn position_sec(&self) -> PyResult<Option<f64>> {
        Ok(self
            .lock_reader()?
            .get_pos()
            .map(|duration| duration.as_secs_f64()))
    }

    #[getter]
    pub fn duration_sec(&self) -> PyResult<Option<f64>> {
        Ok(self
            .lock_reader()?
            .get_duration()
            .map(|duration| duration.as_secs_f64()))
    }
    pub fn grab(&self, py: Python<'_>) -> PyResult<Option<PyImageApi>> {
        let image = self
            .lock_reader()?
            .grab_rgb8()
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;

        let Some(image) = image else {
            return Ok(None);
        };

        let height = image.height();
        let width = image.width();

        let array = unsafe { PyArray::<u8, _>::new(py, [height, width, 3], false) };

        unsafe {
            array
                .as_slice_mut()
                .expect("freshly allocated numpy array should be contiguous")
                .copy_from_slice(image.as_slice());
        }

        Ok(Some(PyImageApi::wrap(
            py,
            array.unbind(),
            Some("RGB".to_string()),
        )))
    }
}
