use numpy::PyArray3;
use pyo3::prelude::*;

use crate::image::{PyImageApi, LUMINANCE_WEIGHTS};

/// Randomly change brightness, contrast, saturation, and hue.
#[pyclass(name = "ColorJitter", module = "kornia_rs")]
pub struct PyColorJitter {
    brightness: (f64, f64),
    contrast: (f64, f64),
    saturation: (f64, f64),
    hue: (f64, f64),
}

fn check_input(value: f64, name: &str, center: f64, bound: Option<(f64, f64)>) -> PyResult<(f64, f64)> {
    if value < 0.0 {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
            "{} must be non-negative, got {}",
            name, value
        )));
    }
    let range = if name == "hue" {
        (-value, value)
    } else {
        (f64::max(0.0, center - value), center + value)
    };
    if let Some((lo, hi)) = bound {
        if range.0 < lo || range.1 > hi {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                "{} values should be between ({}, {})",
                name, lo, hi
            )));
        }
    }
    Ok(range)
}

#[pymethods]
impl PyColorJitter {
    #[new]
    #[pyo3(signature = (brightness=0.0, contrast=0.0, saturation=0.0, hue=0.0))]
    fn new(brightness: f64, contrast: f64, saturation: f64, hue: f64) -> PyResult<Self> {
        Ok(Self {
            brightness: check_input(brightness, "brightness", 1.0, None)?,
            contrast: check_input(contrast, "contrast", 1.0, None)?,
            saturation: check_input(saturation, "saturation", 1.0, None)?,
            hue: check_input(hue, "hue", 0.0, Some((-0.5, 0.5)))?,
        })
    }

    fn __call__(&self, py: Python<'_>, img: PyRef<'_, PyImageApi>) -> PyResult<PyImageApi> {
        let random = py.import("random")?;
        let np = py.import("numpy")?;

        // Build ops list: (op_id, factor)
        // 0=brightness, 1=contrast, 2=saturation, 3=hue
        let mut ops: Vec<(u8, f64)> = Vec::with_capacity(4);

        if self.brightness.0 != 1.0 || self.brightness.1 != 1.0 {
            let factor: f64 = random
                .call_method1("uniform", (self.brightness.0, self.brightness.1))?
                .extract()?;
            ops.push((0, factor));
        }
        if self.contrast.0 != 1.0 || self.contrast.1 != 1.0 {
            let factor: f64 = random
                .call_method1("uniform", (self.contrast.0, self.contrast.1))?
                .extract()?;
            ops.push((1, factor));
        }
        if self.saturation.0 != 1.0 || self.saturation.1 != 1.0 {
            let factor: f64 = random
                .call_method1("uniform", (self.saturation.0, self.saturation.1))?
                .extract()?;
            ops.push((2, factor));
        }
        if self.hue.0 != 0.0 || self.hue.1 != 0.0 {
            let factor: f64 = random
                .call_method1("uniform", (self.hue.0, self.hue.1))?
                .extract()?;
            ops.push((3, factor));
        }

        // Shuffle order (Fisher-Yates using Python's random)
        for i in (1..ops.len()).rev() {
            let j: usize = random
                .call_method1("randint", (0i32, i as i32))?
                .extract()?;
            ops.swap(i, j);
        }

        // Start with a copy of the data as numpy array
        let mut arr: Py<PyArray3<u8>> = img.data(py).bind(py).call_method0("copy")?.extract()?;

        for (op, factor) in &ops {
            match op {
                0 => {
                    // Brightness: multiplicative (torchvision convention)
                    let float_data = arr.bind(py).call_method1("astype", ("float32",))?;
                    let scaled = np.call_method1("multiply", (&float_data, *factor))?;
                    let clipped = np.call_method1("clip", (&scaled, 0.0f64, 255.0f64))?;
                    arr = clipped
                        .call_method1("astype", ("uint8",))?
                        .extract()?;
                }
                1 => {
                    // Contrast
                    let mean: f64 = arr.bind(py).call_method0("mean")?.extract()?;
                    let float_data = arr.bind(py).call_method1("astype", ("float32",))?;
                    let centered = np.call_method1("subtract", (&float_data, mean))?;
                    let scaled = np.call_method1("multiply", (&centered, *factor))?;
                    let shifted = np.call_method1("add", (&scaled, mean))?;
                    let clipped = np.call_method1("clip", (&shifted, 0.0f64, 255.0f64))?;
                    arr = clipped
                        .call_method1("astype", ("uint8",))?
                        .extract()?;
                }
                2 => {
                    // Saturation
                    let float_data = arr.bind(py).call_method1("astype", ("float32",))?;
                    let weights =
                        numpy::PyArray::from_vec(py, LUMINANCE_WEIGHTS.to_vec());
                    let gray = np.call_method1("dot", (&float_data, weights))?;
                    let gray = np.call_method1("expand_dims", (&gray, 2i32))?;
                    let diff = np.call_method1("subtract", (&float_data, &gray))?;
                    let scaled = np.call_method1("multiply", (&diff, *factor))?;
                    let result_f = np.call_method1("add", (&gray, &scaled))?;
                    let clipped =
                        np.call_method1("clip", (&result_f, 0.0f64, 255.0f64))?;
                    arr = clipped
                        .call_method1("astype", ("uint8",))?
                        .extract()?;
                }
                3 => {
                    // Hue (use Image method via a temporary)
                    let tmp = PyImageApi::wrap(py, arr, Some("RGB".to_string()));
                    let result = tmp.adjust_hue(py, *factor)?;
                    arr = result.data(py);
                }
                _ => {}
            }
        }

        Ok(PyImageApi::wrap(py, arr, Some(img.mode().to_string())))
    }

    fn __repr__(&self) -> String {
        format!(
            "ColorJitter(brightness={:?}, contrast={:?}, saturation={:?}, hue={:?})",
            self.brightness, self.contrast, self.saturation, self.hue
        )
    }
}

/// Randomly flip image horizontally with probability p.
#[pyclass(name = "RandomHorizontalFlip", module = "kornia_rs")]
pub struct PyRandomHorizontalFlip {
    p: f64,
}

#[pymethods]
impl PyRandomHorizontalFlip {
    #[new]
    #[pyo3(signature = (p=0.5))]
    fn new(p: f64) -> Self {
        Self { p }
    }

    fn __call__(&self, py: Python<'_>, img: PyRef<'_, PyImageApi>) -> PyResult<PyImageApi> {
        let random = py.import("random")?;
        let val: f64 = random.call_method0("random")?.extract()?;
        if val < self.p {
            img.flip_horizontal(py)
        } else {
            img.copy(py)
        }
    }

    fn __repr__(&self) -> String {
        format!("RandomHorizontalFlip(p={})", self.p)
    }
}

/// Randomly flip image vertically with probability p.
#[pyclass(name = "RandomVerticalFlip", module = "kornia_rs")]
pub struct PyRandomVerticalFlip {
    p: f64,
}

#[pymethods]
impl PyRandomVerticalFlip {
    #[new]
    #[pyo3(signature = (p=0.5))]
    fn new(p: f64) -> Self {
        Self { p }
    }

    fn __call__(&self, py: Python<'_>, img: PyRef<'_, PyImageApi>) -> PyResult<PyImageApi> {
        let random = py.import("random")?;
        let val: f64 = random.call_method0("random")?.extract()?;
        if val < self.p {
            img.flip_vertical(py)
        } else {
            img.copy(py)
        }
    }

    fn __repr__(&self) -> String {
        format!("RandomVerticalFlip(p={})", self.p)
    }
}

/// Randomly crop image to given size.
#[pyclass(name = "RandomCrop", module = "kornia_rs")]
pub struct PyRandomCrop {
    height: usize,
    width: usize,
}

#[pymethods]
impl PyRandomCrop {
    #[new]
    fn new(size: (usize, usize)) -> Self {
        Self {
            height: size.0,
            width: size.1,
        }
    }

    fn __call__(&self, py: Python<'_>, img: PyRef<'_, PyImageApi>) -> PyResult<PyImageApi> {
        let img_w = img.width(py);
        let img_h = img.height(py);

        if img_w < self.width || img_h < self.height {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                "Image ({}x{}) is smaller than crop size ({}x{})",
                img_w, img_h, self.width, self.height
            )));
        }

        let random = py.import("random")?;
        let x: usize = random
            .call_method1("randint", (0i32, (img_w - self.width) as i32))?
            .extract()?;
        let y: usize = random
            .call_method1("randint", (0i32, (img_h - self.height) as i32))?
            .extract()?;

        img.crop(py, x, y, self.width, self.height)
    }

    fn __repr__(&self) -> String {
        format!("RandomCrop(size=({}, {}))", self.height, self.width)
    }
}

/// Randomly rotate image within degree range.
#[pyclass(name = "RandomRotation", module = "kornia_rs")]
pub struct PyRandomRotation {
    degrees: (f64, f64),
}

#[pymethods]
impl PyRandomRotation {
    #[new]
    fn new(degrees: f64) -> Self {
        Self {
            degrees: (-degrees, degrees),
        }
    }

    fn __call__(&self, py: Python<'_>, img: PyRef<'_, PyImageApi>) -> PyResult<PyImageApi> {
        let random = py.import("random")?;
        let angle: f64 = random
            .call_method1("uniform", (self.degrees.0, self.degrees.1))?
            .extract()?;
        img.rotate(py, angle)
    }

    fn __repr__(&self) -> String {
        format!("RandomRotation(degrees={:?})", self.degrees)
    }
}

/// Compose several transforms together.
#[pyclass(name = "Compose", module = "kornia_rs")]
pub struct PyCompose {
    transforms: Vec<Py<PyAny>>,
}

#[pymethods]
impl PyCompose {
    #[new]
    fn new(transforms: Vec<Py<PyAny>>) -> Self {
        Self { transforms }
    }

    fn __call__(&self, py: Python<'_>, img: Py<PyImageApi>) -> PyResult<Py<PyImageApi>> {
        let mut current = img;
        for t in &self.transforms {
            let result = t.call1(py, (&current,))?;
            current = result.extract(py)?;
        }
        Ok(current)
    }

    fn __repr__(&self, py: Python<'_>) -> PyResult<String> {
        let mut lines = Vec::new();
        for t in &self.transforms {
            let repr = t.call_method0(py, "__repr__")?;
            let s: String = repr.extract(py)?;
            lines.push(format!("  {}", s));
        }
        Ok(format!("Compose([\n{}\n])", lines.join(",\n")))
    }
}
