use std::sync::Mutex;

use kornia_io::stream::video::{ImageFormat, VideoReader};
use numpy::{PyArray, PyArray3, PyArrayMethods};
use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;

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

#[pymethods]
impl PyVideoReader {
    #[new]
    pub fn new(path: &str, format: PyImageFormat) -> PyResult<Self> {
        let reader = VideoReader::new(path, format.into())
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;

        Ok(Self {
            reader: Mutex::new(reader),
            format,
        })
    }

    pub fn start(&self) -> PyResult<()> {
        self.reader
            .lock()
            .unwrap()
            .start()
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))
    }

    pub fn pause(&self) -> PyResult<()> {
        self.reader
            .lock()
            .unwrap()
            .pause()
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))
    }

    pub fn close(&self) -> PyResult<()> {
        self.reader
            .lock()
            .unwrap()
            .close()
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))
    }

    pub fn reset(&self) -> PyResult<()> {
        self.reader
            .lock()
            .unwrap()
            .reset()
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))
    }

    #[getter]
    pub fn fps(&self) -> Option<f64> {
        self.reader.lock().unwrap().get_fps()
    }

    #[getter]
    pub fn position_sec(&self) -> Option<f64> {
        self.reader
            .lock()
            .unwrap()
            .get_pos()
            .map(|duration| duration.as_secs_f64())
    }

    #[getter]
    pub fn duration_sec(&self) -> Option<f64> {
        self.reader
            .lock()
            .unwrap()
            .get_duration()
            .map(|duration| duration.as_secs_f64())
    }

    pub fn grab(&self, py: Python<'_>) -> PyResult<Option<Py<PyArray3<u8>>>> {
        match self.format {
            PyImageFormat::Rgb8 => {
                let image = self
                    .reader
                    .lock()
                    .unwrap()
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

                Ok(Some(array.unbind()))
            }
            PyImageFormat::Mono8 => Err(PyRuntimeError::new_err(
                "Mono8 video grab is not supported yet",
            )),
        }
    }
}
