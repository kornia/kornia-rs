use pyo3::{
    pymodule,
    types::{PyModule, PyModuleMethods},
    Bound, PyResult,
};

pub fn apriltag(m: &Bound<'_, PyModule>) -> PyResult<()> {
    let family_mod = PyModule::new(m.py(), "family")?;
    family::init(&family_mod)?;
    m.add_submodule(&family_mod)?;

    Ok(())
}

#[pymodule]
mod family {
    use kornia_apriltag::family::TagFamilyKind;
    use pyo3::{exceptions::PyException, prelude::*, types::PyList, PyResult};

    #[pymodule_init]
    pub fn init(m: &Bound<'_, PyModule>) -> PyResult<()> {
        m.add_class::<PyTagFamilyKind>()?;
        m.add_function(wrap_pyfunction!(all_tag_family_kind, m)?)?;
        Ok(())
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
            match self.0 {
                TagFamilyKind::Tag16H5 => Ok("tag16_h5"),
                TagFamilyKind::Tag36H11 => Ok("tag36_h11"),
                TagFamilyKind::Tag36H10 => Ok("tag36_h10"),
                TagFamilyKind::Tag25H9 => Ok("tag25_h9"),
                TagFamilyKind::TagCircle21H7 => Ok("tagcircle21_h7"),
                TagFamilyKind::TagCircle49H12 => Ok("tagcircle49_h12"),
                TagFamilyKind::TagCustom48H12 => Ok("tagcustom48_h12"),
                TagFamilyKind::TagStandard41H12 => Ok("tagstandard41_h12"),
                TagFamilyKind::TagStandard52H13 => Ok("tagstandard52_h13"),
                _ => Err(PyErr::new::<PyException, _>("unknown tagfamily kind")),
            }
        }
    }

    #[pyfunction]
    pub fn all_tag_family_kind() -> PyResult<Py<PyList>> {
        let all = TagFamilyKind::all();
        let py_all: Vec<_> = all.iter().map(|tag| PyTagFamilyKind(tag.clone())).collect();

        Python::attach(|py| match PyList::new(py, py_all) {
            Ok(list) => Ok(list.unbind()),
            Err(e) => Err(e),
        })
    }
}
