use numpy::PyArray3;
use pyo3::prelude::*;
use rand::prelude::*;
use rand::rngs::StdRng;
use std::cell::RefCell;
use std::sync::Mutex;

use crate::image::{PyImageApi, LUMINANCE_WEIGHTS};

// Global seed. When set, each augmentation call creates a seeded RNG.
// When None, thread_rng() is used.
static GLOBAL_SEED: Mutex<Option<u64>> = Mutex::new(None);

thread_local! {
    static SEEDED_RNG: RefCell<Option<StdRng>> = const { RefCell::new(None) };
}

fn with_rng<T>(f: impl FnOnce(&mut dyn RngCore) -> T) -> T {
    // Check if a global seed was set and we need to initialize
    let seed = GLOBAL_SEED.lock().unwrap();
    if let Some(s) = *seed {
        drop(seed);
        SEEDED_RNG.with(|cell| {
            let mut rng_opt = cell.borrow_mut();
            if rng_opt.is_none() {
                *rng_opt = Some(StdRng::seed_from_u64(s));
            }
            f(rng_opt.as_mut().unwrap())
        })
    } else {
        drop(seed);
        f(&mut rand::rng())
    }
}

/// Set the random seed for all augmentation operations.
///
/// After calling this, augmentations will produce reproducible results.
/// Pass None to reset to non-deterministic mode.
#[pyfunction]
#[pyo3(signature = (seed=None))]
pub fn set_seed(seed: Option<u64>) {
    let mut global = GLOBAL_SEED.lock().unwrap();
    *global = seed;
    // Reset thread-local RNG so it picks up the new seed
    SEEDED_RNG.with(|cell| {
        let mut rng_opt = cell.borrow_mut();
        if let Some(s) = seed {
            *rng_opt = Some(StdRng::seed_from_u64(s));
        } else {
            *rng_opt = None;
        }
    });
}

/// Randomly change brightness, contrast, saturation, and hue.
#[pyclass(name = "ColorJitter", module = "kornia_rs")]
pub struct PyColorJitter {
    brightness: (f64, f64),
    contrast: (f64, f64),
    saturation: (f64, f64),
    hue: (f64, f64),
}

fn check_input(
    value: f64,
    name: &str,
    center: f64,
    bound: Option<(f64, f64)>,
) -> PyResult<(f64, f64)> {
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
        let np = py.import("numpy")?;

        // Build ops list and shuffle using Rust RNG
        let mut ops: Vec<(u8, f64)> = Vec::with_capacity(4);

        with_rng(|rng| {
            if self.brightness.0 != 1.0 || self.brightness.1 != 1.0 {
                let factor = rng.random_range(self.brightness.0..=self.brightness.1);
                ops.push((0, factor));
            }
            if self.contrast.0 != 1.0 || self.contrast.1 != 1.0 {
                let factor = rng.random_range(self.contrast.0..=self.contrast.1);
                ops.push((1, factor));
            }
            if self.saturation.0 != 1.0 || self.saturation.1 != 1.0 {
                let factor = rng.random_range(self.saturation.0..=self.saturation.1);
                ops.push((2, factor));
            }
            if self.hue.0 != 0.0 || self.hue.1 != 0.0 {
                let factor = rng.random_range(self.hue.0..=self.hue.1);
                ops.push((3, factor));
            }
            ops.shuffle(rng);
        });

        // Start with a copy of the data as numpy array
        let mut arr: Py<PyArray3<u8>> = img.data(py).bind(py).call_method0("copy")?.extract()?;

        for (op, factor) in &ops {
            match op {
                0 => {
                    // Brightness: multiplicative (torchvision convention)
                    let float_data = arr.bind(py).call_method1("astype", ("float32",))?;
                    let scaled = np.call_method1("multiply", (&float_data, *factor))?;
                    let clipped = np.call_method1("clip", (&scaled, 0.0f64, 255.0f64))?;
                    arr = clipped.call_method1("astype", ("uint8",))?.extract()?;
                }
                1 => {
                    // Contrast
                    let mean: f64 = arr.bind(py).call_method0("mean")?.extract()?;
                    let float_data = arr.bind(py).call_method1("astype", ("float32",))?;
                    let centered = np.call_method1("subtract", (&float_data, mean))?;
                    let scaled = np.call_method1("multiply", (&centered, *factor))?;
                    let shifted = np.call_method1("add", (&scaled, mean))?;
                    let clipped = np.call_method1("clip", (&shifted, 0.0f64, 255.0f64))?;
                    arr = clipped.call_method1("astype", ("uint8",))?.extract()?;
                }
                2 => {
                    // Saturation
                    let float_data = arr.bind(py).call_method1("astype", ("float32",))?;
                    let weights = numpy::PyArray::from_vec(py, LUMINANCE_WEIGHTS.to_vec());
                    let gray = np.call_method1("dot", (&float_data, weights))?;
                    let gray = np.call_method1("expand_dims", (&gray, 2i32))?;
                    let diff = np.call_method1("subtract", (&float_data, &gray))?;
                    let scaled = np.call_method1("multiply", (&diff, *factor))?;
                    let result_f = np.call_method1("add", (&gray, &scaled))?;
                    let clipped = np.call_method1("clip", (&result_f, 0.0f64, 255.0f64))?;
                    arr = clipped.call_method1("astype", ("uint8",))?.extract()?;
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
        let val: f64 = with_rng(|rng| rng.random());
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
        let val: f64 = with_rng(|rng| rng.random());
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

        let (x, y) = with_rng(|rng| {
            let x = rng.random_range(0..=(img_w - self.width));
            let y = rng.random_range(0..=(img_h - self.height));
            (x, y)
        });

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
        let angle: f64 = with_rng(|rng| rng.random_range(self.degrees.0..=self.degrees.1));
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
