use kornia_apriltag::{decoder::Detection, quad::Quad, AprilTagDecoder, DecodeTagsConfig};
use kornia_image::Image;
use pyo3::{
    exceptions::PyException,
    prelude::*,
    types::{PyModule, PyModuleMethods},
    Bound, PyResult,
};

use crate::image::{FromPyImage, PyImage, PyImageSize};

#[pyclass(name = "DecodeTagsConfig")]
pub struct PyDecodeTagsConfig(DecodeTagsConfig);

#[pymethods]
impl PyDecodeTagsConfig {
    #[new]
    pub fn new(tag_family_kinds: Vec<Py<family::PyTagFamilyKind>>) -> PyResult<Self> {
        Python::attach(|py| {
            let mut tag_families = Vec::with_capacity(tag_family_kinds.len());
            for family_kind in tag_family_kinds.iter() {
                let py_family_kind: family::PyTagFamilyKind = family_kind.extract(py)?;
                let family = py_family_kind.0;
                tag_families.push(family);
            }

            Ok(Self(DecodeTagsConfig::new(tag_families)))
        })
    }

    #[staticmethod]
    pub fn all() -> Self {
        Self(DecodeTagsConfig::all())
    }
}

#[pyclass(name = "FitQuadConfig", eq, get_all, set_all)]
#[derive(Default, PartialEq, Clone)]
pub struct PyFitQuadConfig {
    pub cos_critical_rad: f32,
    pub max_line_fit_mse: f32,
    pub max_nmaxima: usize,
    pub min_cluster_pixels: usize,
}

#[pyclass(name = "AprilTagDecoder")]
pub struct PyAprilTagDecoder(AprilTagDecoder);

#[pymethods]
impl PyAprilTagDecoder {
    #[new]
    pub fn new(config: Py<PyDecodeTagsConfig>, img_size: PyImageSize) -> PyResult<Self> {
        Python::attach(|py| {
            let config = config.borrow(py);

            Ok(Self(
                AprilTagDecoder::new(config.0.clone(), img_size.into())
                    .map_err(|err| PyErr::new::<PyException, _>(err.to_string()))?,
            ))
        })
    }

    pub fn clear(&mut self) {
        self.0.clear();
    }

    pub fn decode(&mut self, src: PyImage) -> PyResult<Vec<PyDetection>> {
        let img: Image<u8, 1, _> = Image::from_pyimage(src)
            .map_err(|err| PyErr::new::<PyException, _>(err.to_string()))?;

        let detection = self
            .0
            .decode(&img)
            .map_err(|err| PyErr::new::<PyException, _>(err.to_string()))?;

        let py_detection = detection.iter().map(|d| d.clone().into()).collect();
        Ok(py_detection)
    }
}

#[pyclass(name = "Detection", eq, get_all, set_all)]
#[derive(PartialEq, Clone)]
pub struct PyDetection {
    pub tag_family_kind: family::PyTagFamilyKind,
    pub id: u16,
    pub hamming: u8,
    pub decision_margin: f32,
    pub center: (f32, f32),
    pub quad: PyQuad,
}

impl From<Detection> for PyDetection {
    fn from(value: Detection) -> Self {
        Self {
            tag_family_kind: family::PyTagFamilyKind(value.tag_family_kind),
            id: value.id,
            hamming: value.hamming,
            decision_margin: value.decision_margin,
            center: (value.center.x, value.center.y),
            quad: value.quad.into(),
        }
    }
}

#[pyclass(name = "Quad", eq, get_all, set_all)]
#[derive(PartialEq, Clone)]
pub struct PyQuad {
    pub corners: [(f32, f32); 4],
    pub reversed_border: bool,
    pub homography: [f32; 9],
}

impl From<Quad> for PyQuad {
    fn from(value: Quad) -> Self {
        Self {
            corners: [
                (value.corners[0].x, value.corners[0].y),
                (value.corners[1].x, value.corners[1].y),
                (value.corners[2].x, value.corners[2].y),
                (value.corners[3].x, value.corners[3].y),
            ],
            reversed_border: value.reversed_border,
            homography: value.homography,
        }
    }
}

#[pymodule]
pub mod family {
    use kornia_apriltag::{
        decoder::{QuickDecode, SharpeningBuffer},
        family::{TagFamily, TagFamilyKind},
    };
    use pyo3::{exceptions::PyException, prelude::*, Py, PyResult};

    #[pyclass(name = "TagFamily", get_all, set_all)]
    pub struct PyTagFamily {
        pub name: String,
        pub width_at_border: usize,
        pub reversed_border: bool,
        pub total_width: usize,
        pub nbits: usize,
        pub bit_x: Vec<i8>,
        pub bit_y: Vec<i8>,
        pub code_data: Vec<usize>,
        pub quick_decode: Py<PyQuickDecode>,
        pub sharpening_buffer: Py<PySharpeningBuffer>,
    }

    impl PyTagFamily {
        pub fn from_tag_family(py: Python<'_>, family: TagFamily) -> PyResult<Self> {
            Ok(PyTagFamily {
                name: family.name,
                width_at_border: family.width_at_border,
                reversed_border: family.reversed_border,
                total_width: family.total_width,
                nbits: family.nbits,
                bit_x: family.bit_x,
                bit_y: family.bit_y,
                code_data: family.code_data,
                quick_decode: Py::new(py, PyQuickDecode(family.quick_decode))?,
                sharpening_buffer: Py::new(py, PySharpeningBuffer(family.sharpening_buffer))?,
            })
        }
    }

    #[pymethods]
    impl PyTagFamily {
        #[new]
        #[allow(clippy::too_many_arguments)]
        pub fn new(
            name: String,
            width_at_border: usize,
            reversed_border: bool,
            total_width: usize,
            nbits: usize,
            bit_x: Vec<i8>,
            bit_y: Vec<i8>,
            code_data: Vec<usize>,
            quick_decode: Py<PyQuickDecode>,
            sharpening_buffer: Py<PySharpeningBuffer>,
        ) -> Self {
            Self {
                name,
                width_at_border,
                reversed_border,
                total_width,
                nbits,
                bit_x,
                bit_y,
                code_data,
                quick_decode,
                sharpening_buffer,
            }
        }
    }

    #[pyclass(name = "QuickDecode")]
    #[derive(Clone)]
    pub struct PyQuickDecode(pub QuickDecode);

    #[pymethods]
    impl PyQuickDecode {
        #[new]
        pub fn new(nbits: usize, code_data: Vec<usize>) -> Self {
            Self(QuickDecode::new(nbits, &code_data))
        }
    }

    #[pyclass(name = "SharpeningBuffer")]
    #[derive(Clone)]
    pub struct PySharpeningBuffer(pub SharpeningBuffer);

    #[pymethods]
    impl PySharpeningBuffer {
        #[new]
        pub fn new(len: usize) -> Self {
            Self(SharpeningBuffer::new(len))
        }

        pub fn reset(&mut self) {
            self.0.reset();
        }
    }

    #[pyclass(name = "TagFamilyKind", eq)]
    #[derive(PartialEq, Clone)]
    pub struct PyTagFamilyKind(pub TagFamilyKind);

    #[pymethods]
    impl PyTagFamilyKind {
        #[new]
        pub fn new(name: &str) -> PyResult<Self> {
            match name {
                "tag16_h5" => Ok(Self(TagFamilyKind::Tag16H5)),
                "tag36_h11" => Ok(Self(TagFamilyKind::Tag36H11)),
                "tag36_h10" => Ok(Self(TagFamilyKind::Tag36H10)),
                "tag25_h9" => Ok(Self(TagFamilyKind::Tag25H9)),
                "tagcircle21_h7" => Ok(Self(TagFamilyKind::TagCircle21H7)),
                "tagcircle49_h12" => Ok(Self(TagFamilyKind::TagCircle49H12)),
                "tagcustom48_h12" => Ok(Self(TagFamilyKind::TagCustom48H12)),
                "tagstandard41_h12" => Ok(Self(TagFamilyKind::TagStandard41H12)),
                "tagstandard52_h13" => Ok(Self(TagFamilyKind::TagStandard52H13)),
                _ => Err(PyErr::new::<PyException, _>("unsupported tag family kind")),
            }
        }

        #[getter]
        pub fn name(&self) -> PyResult<&str> {
            match &self.0 {
                TagFamilyKind::Tag16H5 => Ok("tag16_h5"),
                TagFamilyKind::Tag36H11 => Ok("tag36_h11"),
                TagFamilyKind::Tag36H10 => Ok("tag36_h10"),
                TagFamilyKind::Tag25H9 => Ok("tag25_h9"),
                TagFamilyKind::TagCircle21H7 => Ok("tagcircle21_h7"),
                TagFamilyKind::TagCircle49H12 => Ok("tagcircle49_h12"),
                TagFamilyKind::TagCustom48H12 => Ok("tagcustom48_h12"),
                TagFamilyKind::TagStandard41H12 => Ok("tagstandard41_h12"),
                TagFamilyKind::TagStandard52H13 => Ok("tagstandard52_h13"),
                TagFamilyKind::Custom(family) => Ok(family.name.as_str()),
            }
        }

        #[staticmethod]
        pub fn all() -> PyResult<Vec<Py<PyTagFamilyKind>>> {
            let all = TagFamilyKind::all();
            let mut py_all = Vec::with_capacity(all.len());

            Python::attach(|py| {
                for kind in all {
                    let py_kind = PyTagFamilyKind(kind);
                    py_all.push(Py::new(py, py_kind)?);
                }

                Ok(py_all)
            })
        }
    }
}
