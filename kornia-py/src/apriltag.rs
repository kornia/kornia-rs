use kornia_apriltag::{
    decoder::Detection,
    quad::{FitQuadConfig, Quad},
    AprilTagDecoder, DecodeTagsConfig,
};
use kornia_image::Image;
use pyo3::{exceptions::PyException, prelude::*, PyResult};

use crate::image::{FromPyImage, PyImage, PyImageSize};

#[pyclass(name = "DecodeTagsConfig")]
pub struct PyDecodeTagsConfig(DecodeTagsConfig);

#[pymethods]
impl PyDecodeTagsConfig {
    #[new]
    #[pyo3(signature = (tag_family_kinds=None))]
    pub fn new(tag_family_kinds: Option<Vec<family::PyTagFamilyKind>>) -> PyResult<Self> {
        let kinds = tag_family_kinds.unwrap_or_default();
        let mut tag_families = Vec::with_capacity(kinds.len());
        for py_kind in kinds {
            tag_families.push(py_kind.0);
        }
        Ok(Self(DecodeTagsConfig::new(tag_families).map_err(|e| {
            PyErr::new::<PyException, _>(e.to_string())
        })?))
    }

    pub fn add(&mut self, family_kind: family::PyTagFamilyKind) {
        self.0.add(family_kind.0);
    }

    // TODO: Add getter for tag_families

    #[getter]
    pub fn get_fit_quad_config(&self) -> PyFitQuadConfig {
        PyFitQuadConfig {
            cos_critical_rad: self.0.fit_quad_config.cos_critical_rad,
            max_line_fit_mse: self.0.fit_quad_config.max_line_fit_mse,
            max_nmaxima: self.0.fit_quad_config.max_nmaxima,
            min_cluster_pixels: self.0.fit_quad_config.min_cluster_pixels,
        }
    }

    #[setter]
    pub fn set_fit_quad_config(&mut self, value: PyFitQuadConfig) {
        let config = FitQuadConfig {
            cos_critical_rad: value.cos_critical_rad,
            max_line_fit_mse: value.max_line_fit_mse,
            max_nmaxima: value.max_nmaxima,
            min_cluster_pixels: value.min_cluster_pixels,
        };

        self.0.fit_quad_config = config;
    }

    #[getter]
    pub fn get_refine_edges_enabled(&self) -> bool {
        self.0.refine_edges_enabled
    }

    #[setter]
    pub fn set_refine_edges_enabled(&mut self, value: bool) {
        self.0.refine_edges_enabled = value;
    }

    #[getter]
    pub fn get_decode_sharpening(&self) -> f32 {
        self.0.decode_sharpening
    }

    #[setter]
    pub fn set_decode_sharpening(&mut self, value: f32) {
        self.0.decode_sharpening = value;
    }

    #[getter]
    pub fn get_normal_border(&self) -> bool {
        self.0.normal_border
    }

    #[setter]
    pub fn set_normal_border(&mut self, value: bool) {
        self.0.normal_border = value;
    }

    #[getter]
    pub fn get_reversed_border(&self) -> bool {
        self.0.reversed_border
    }

    #[setter]
    pub fn set_reversed_border(&mut self, value: bool) {
        self.0.reversed_border = value;
    }

    #[getter]
    pub fn get_min_tag_width(&self) -> usize {
        self.0.min_tag_width
    }

    #[setter]
    pub fn set_min_tag_width(&mut self, value: usize) {
        self.0.min_tag_width = value;
    }

    #[getter]
    pub fn get_min_white_black_difference(&self) -> u8 {
        self.0.min_white_black_difference
    }

    #[setter]
    pub fn set_min_white_black_difference(&mut self, value: u8) {
        self.0.min_white_black_difference = value;
    }

    #[getter]
    pub fn get_downscale_factor(&self) -> usize {
        self.0.downscale_factor
    }

    #[setter]
    pub fn set_downscale_factor(&mut self, value: usize) {
        self.0.downscale_factor = value;
    }

    #[staticmethod]
    pub fn all() -> PyResult<Self> {
        Ok(Self(DecodeTagsConfig::all().map_err(|e| {
            PyErr::new::<PyException, _>(e.to_string())
        })?))
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

#[pymethods]
impl PyFitQuadConfig {
    #[new]
    pub fn new(
        cos_critical_rad: f32,
        max_line_fit_mse: f32,
        max_nmaxima: usize,
        min_cluster_pixels: usize,
    ) -> Self {
        Self {
            cos_critical_rad,
            max_line_fit_mse,
            max_nmaxima,
            min_cluster_pixels,
        }
    }
}

#[pyclass(name = "AprilTagDecoder")]
/// AprilTag detector for identifying and decoding AprilTags in images.
///
/// The decoder is stateful and maintains internal buffers for performance.
/// It expects single-channel 8-bit images (grayscale).
///
/// # Arguments
/// * `config` - The decoding configuration.
/// * `img_size` - The size of images to be processed. All input images must match this size.
///
/// # Returns
/// * A list of `ApriltagDetection` objects.
///
/// # Exceptions
/// Raises `PyException` if:
/// * The input image cannot be converted to grayscale u8.
/// * The input image size does not match the configured size.
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

    pub fn decode(&mut self, src: PyImage) -> PyResult<Vec<PyApriltagDetection>> {
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

#[pyclass(name = "ApriltagDetection", eq, get_all, set_all)]
#[derive(PartialEq, Clone)]
pub struct PyApriltagDetection {
    pub tag_family_kind: family::PyTagFamilyKind,
    pub id: u16,
    pub hamming: u8,
    pub decision_margin: f32,
    pub center: (f32, f32),
    pub quad: PyQuad,
}

#[pymethods]
impl PyApriltagDetection {
    #[new]
    pub fn new(
        tag_family_kind: family::PyTagFamilyKind,
        id: u16,
        hamming: u8,
        decision_margin: f32,
        center: (f32, f32),
        quad: PyQuad,
    ) -> Self {
        Self {
            tag_family_kind,
            id,
            hamming,
            decision_margin,
            center,
            quad,
        }
    }
}

impl From<Detection> for PyApriltagDetection {
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

#[pymethods]
impl PyQuad {
    #[new]
    pub fn new(corners: [(f32, f32); 4], reversed_border: bool, homography: [f32; 9]) -> Self {
        Self {
            corners,
            reversed_border,
            homography,
        }
    }
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
            homography: value.homography.into(),
        }
    }
}

pub mod family {
    use kornia_apriltag::{
        decoder::{QuickDecode, SharpeningBuffer},
        family::{TagFamily, TagFamilyKind},
    };
    use pyo3::{exceptions::PyException, prelude::*, Py, PyResult};
    use std::sync::Arc;

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
        /// The minimum Hamming distance between any two valid codes in this family.
        pub min_hamming: u8,
        pub quick_decode: Py<PyQuickDecode>,
        pub sharpening_buffer: Py<PySharpeningBuffer>,
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
            min_hamming: u8,
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
                min_hamming,
                quick_decode,
                sharpening_buffer,
            }
        }

        #[allow(clippy::wrong_self_convention)]
        pub fn into_tag_family_kind(&self) -> PyResult<PyTagFamilyKind> {
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
                    min_hamming: self.min_hamming,
                    quick_decode: quick_decode.0,
                    sharpening_buffer: sharpening_buffer.0,
                };

                let kind = TagFamilyKind::Custom(Arc::new(tag_family));
                Ok(PyTagFamilyKind(kind))
            })
        }
    }

    #[pyclass(name = "QuickDecode")]
    #[derive(Clone)]
    pub struct PyQuickDecode(pub QuickDecode);

    #[pymethods]
    impl PyQuickDecode {
        #[new]
        #[pyo3(signature = (nbits, code_data, max_hamming=2))]
        pub fn new(nbits: usize, code_data: Vec<usize>, max_hamming: u8) -> PyResult<Self> {
            Ok(Self(
                QuickDecode::new(nbits, &code_data, max_hamming)
                    .map_err(|e| PyErr::new::<PyException, _>(e.to_string()))?,
            ))
        }

        #[getter]
        pub fn get_max_hamming(&self) -> u8 {
            self.0.max_hamming()
        }

        #[setter]
        pub fn set_max_hamming(&mut self, value: u8) -> PyResult<()> {
            self.0
                .set_max_hamming(value)
                .map_err(|e| PyErr::new::<PyException, _>(e.to_string()))
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

        /// The minimum Hamming distance between any two valid codes in this family.
        #[getter]
        pub fn min_hamming(&self) -> PyResult<u8> {
            let family: TagFamily = self.0.clone().try_into().map_err(
                |e: kornia_apriltag::errors::AprilTagError| {
                    PyErr::new::<PyException, _>(e.to_string())
                },
            )?;
            Ok(family.min_hamming)
        }

        /// The maximum safe max_hamming value for this family.
        ///
        /// Calculated as (min_hamming - 1) / 2 to ensure two different tags
        /// with bit errors cannot be confused with each other.
        #[getter]
        pub fn max_safe_hamming(&self) -> PyResult<u8> {
            let family: TagFamily = self.0.clone().try_into().map_err(
                |e: kornia_apriltag::errors::AprilTagError| {
                    PyErr::new::<PyException, _>(e.to_string())
                },
            )?;
            Ok(family.max_safe_hamming())
        }

        /// Returns a new TagFamilyKind with the specified max_hamming value.
        ///
        /// Raises an exception if max_hamming exceeds the safe limit for this family.
        pub fn with_max_hamming(&self, max_hamming: u8) -> PyResult<Self> {
            let family: TagFamily = self.0.clone().try_into().map_err(
                |e: kornia_apriltag::errors::AprilTagError| {
                    PyErr::new::<PyException, _>(e.to_string())
                },
            )?;
            let updated = family
                .with_max_hamming(max_hamming)
                .map_err(|e| PyErr::new::<PyException, _>(e.to_string()))?;
            Ok(Self(TagFamilyKind::Custom(Arc::new(updated))))
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

#[pymodule]
fn apriltag(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyAprilTagDecoder>()?;
    m.add_class::<PyApriltagDetection>()?;
    m.add_class::<PyDecodeTagsConfig>()?;
    m.add_class::<PyFitQuadConfig>()?;

    m.add_class::<family::PyTagFamilyKind>()?;

    Ok(())
}
