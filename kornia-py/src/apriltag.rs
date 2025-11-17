use pyo3::{
    pymodule,
    types::{PyModule, PyModuleMethods},
    Bound, PyResult,
};

pub fn init(m: &Bound<'_, PyModule>) -> PyResult<()> {
    let family_mod = PyModule::new(m.py(), "family")?;
    family::init(&family_mod)?;
    m.add_submodule(&family_mod)?;

    Ok(())
}

#[pymodule]
mod family {
    use kornia_apriltag::{
        decoder::{QuickDecode, SharpeningBuffer},
        family::{TagFamily, TagFamilyKind},
    };
    use pyo3::{exceptions::PyException, prelude::*, types::PyList, Py, PyResult};

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
    pub struct PyQuickDecode(QuickDecode);

    #[pymethods]
    impl PyQuickDecode {
        #[new]
        pub fn new(nbits: usize, code_data: Vec<usize>) -> Self {
            Self(QuickDecode::new(nbits, &code_data))
        }
    }

    #[pyclass(name = "SharpeningBuffer")]
    #[derive(Clone)]
    pub struct PySharpeningBuffer(SharpeningBuffer);

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
    #[derive(PartialEq)]
    pub struct PyTagFamilyKind(TagFamilyKind);

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
        pub fn all() -> PyResult<Py<PyList>> {
            let all = TagFamilyKind::all();
            let py_all: Vec<_> = all.iter().map(|tag| PyTagFamilyKind(tag.clone())).collect();

            Python::attach(|py| match PyList::new(py, py_all) {
                Ok(list) => Ok(list.unbind()),
                Err(e) => Err(e),
            })
        }
    }
}
