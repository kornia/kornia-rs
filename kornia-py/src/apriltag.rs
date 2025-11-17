use kornia_apriltag::{
    decoder::Detection,
    family::TagFamily,
    quad::{FitQuadConfig, Quad},
    AprilTagDecoder, DecodeTagsConfig,
};
use kornia_image::Image;
use pyo3::{
    exceptions::PyException,
    prelude::*,
    types::{PyModule, PyModuleMethods},
    Bound, PyResult,
};

use crate::{
    apriltag::family::PyTagFamilyKind,
    image::{FromPyImage, PyImage, PyImageSize},
};

pub fn init(m: &Bound<'_, PyModule>) -> PyResult<()> {
    let family_mod = PyModule::new(m.py(), "family")?;
    family::init(&family_mod)?;
    m.add_submodule(&family_mod)?;

    m.add_class::<PyDecodeTagsConfig>()?;
    m.add_class::<PyFitQuadConfig>()?;
    m.add_class::<PyAprilTagDecoder>()?;
    m.add_class::<PyDetection>()?;
    m.add_class::<PyQuad>()?;

    Ok(())
}

#[pyclass(name = "DecodeTagsConfig", get_all, set_all)]
#[derive(Default)]
pub struct PyDecodeTagsConfig {
    pub tag_families: Vec<Py<family::PyTagFamily>>,
    pub fit_quad_config: PyFitQuadConfig,
    pub refine_edges_enabled: bool,
    pub decode_sharpening: f32,
    pub normal_border: bool,
    pub reversed_border: bool,
    pub min_tag_width: usize,
    pub min_white_black_difference: u8,
    pub downscale_factor: usize,
}

impl PyDecodeTagsConfig {
    pub fn into_decode_tags_config(self_: Py<Self>, py: Python<'_>) -> PyResult<DecodeTagsConfig> {
        let self_: PyRef<Self> = self_.extract(py)?;
        let mut tag_families = Vec::with_capacity(self_.tag_families.len());

        for tag in &self_.tag_families {
            let py_tag: PyRef<family::PyTagFamily> = tag.extract(py)?;
            tag_families.push(TagFamily {
                name: py_tag.name.clone(),
                width_at_border: py_tag.width_at_border,
                reversed_border: py_tag.reversed_border,
                total_width: py_tag.total_width,
                nbits: py_tag.nbits,
                bit_x: py_tag.bit_x.clone(),
                bit_y: py_tag.bit_y.clone(),
                code_data: py_tag.code_data.clone(),
                quick_decode: py_tag.quick_decode.extract::<family::PyQuickDecode>(py)?.0,
                sharpening_buffer: py_tag
                    .sharpening_buffer
                    .extract::<family::PySharpeningBuffer>(py)?
                    .0,
            });
        }

        let fit_quad_config = FitQuadConfig {
            cos_critical_rad: self_.fit_quad_config.cos_critical_rad,
            max_line_fit_mse: self_.fit_quad_config.max_line_fit_mse,
            max_nmaxima: self_.fit_quad_config.max_nmaxima,
            min_cluster_pixels: self_.fit_quad_config.min_cluster_pixels,
        };

        Ok(DecodeTagsConfig {
            tag_families,
            fit_quad_config,
            refine_edges_enabled: self_.refine_edges_enabled,
            decode_sharpening: self_.decode_sharpening,
            normal_border: self_.normal_border,
            reversed_border: self_.reversed_border,
            min_tag_width: self_.min_tag_width,
            min_white_black_difference: self_.min_white_black_difference,
            downscale_factor: self_.downscale_factor,
        })
    }
}

#[pymethods]
impl PyDecodeTagsConfig {
    #[new]
    pub fn new(tag_family_kinds: Vec<Py<family::PyTagFamilyKind>>) -> PyResult<Self> {
        Python::attach(|py| {
            let mut tag_families = Vec::with_capacity(tag_family_kinds.len());
            for family_kind in tag_family_kinds.iter() {
                let py_family_kind: family::PyTagFamilyKind = family_kind.extract(py)?;
                let py_family = py_family_kind.into_family()?;
                tag_families.push(Py::new(py, py_family)?);
            }

            Ok(PyDecodeTagsConfig {
                tag_families,
                ..Default::default()
            })
        })
    }

    pub fn add(&mut self, family: Py<family::PyTagFamily>) {
        self.tag_families.push(family);
    }

    #[staticmethod]
    pub fn all() -> PyResult<Self> {
        let kinds = PyTagFamilyKind::all()?;
        Self::new(kinds)
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
            let config = PyDecodeTagsConfig::into_decode_tags_config(config, py)?;

            Ok(Self(
                AprilTagDecoder::new(config, img_size.into())
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
mod family {
    use kornia_apriltag::{
        decoder::{QuickDecode, SharpeningBuffer},
        family::{TagFamily, TagFamilyKind},
    };
    use pyo3::{exceptions::PyException, prelude::*, Py, PyResult};

    #[pymodule_init]
    pub fn init(m: &Bound<'_, PyModule>) -> PyResult<()> {
        m.add_class::<PyTagFamily>()?;
        m.add_class::<PyTagFamilyKind>()?;
        m.add_class::<PyQuickDecode>()?;
        m.add_class::<PySharpeningBuffer>()?;
        Ok(())
    }

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

        #[allow(clippy::wrong_self_convention)]
        pub fn into_family_kind(&self) -> PyResult<Py<PyTagFamilyKind>> {
            Python::attach(|py| {
                let quick_decode: PyQuickDecode = self.quick_decode.extract(py)?;
                let sharpening_buffer: PySharpeningBuffer = self.sharpening_buffer.extract(py)?;

                let tag_family = TagFamily {
                    name: self.name.clone(),
                    width_at_border: self.width_at_border,
                    reversed_border: self.reversed_border,
                    total_width: self.total_width,
                    nbits: self.nbits,
                    bit_x: self.bit_x.clone(),
                    bit_y: self.bit_y.clone(),
                    code_data: self.code_data.clone(),
                    quick_decode: quick_decode.0,
                    sharpening_buffer: sharpening_buffer.0,
                };

                let kind = TagFamilyKind::Custom(tag_family);
                Py::new(py, PyTagFamilyKind(kind))
            })
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

        pub fn into_family(&self) -> PyResult<PyTagFamily> {
            let inner: TagFamily = self.0.clone().into();
            Python::attach(|py| Ok(PyTagFamily::from_tag_family(py, inner)?))
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
