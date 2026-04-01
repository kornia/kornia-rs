use pyo3::prelude::*;
use rand::prelude::*;
use rand::rngs::StdRng;
use std::cell::RefCell;
use std::sync::Mutex;

use crate::image::{pyarray_data, vec_to_pyarray, PyImageApi};

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
#[pyclass(name = "ColorJitter")]
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

        let mut current = img.copy(py)?;

        for (op, factor) in &ops {
            current = match op {
                0 => {
                    // Brightness: multiplicative (torchvision convention)
                    let arr = current.data(py);
                    let bound = arr.bind(py);
                    let (src, h, w, c) = pyarray_data(&bound);
                    let out: Vec<u8> = src
                        .iter()
                        .map(|&v| (v as f64 * factor).clamp(0.0, 255.0) as u8)
                        .collect();
                    PyImageApi::wrap(py, vec_to_pyarray(py, out, h, w, c), Some(current.mode().to_string()))
                }
                1 => current.adjust_contrast(py, *factor)?,
                2 => current.adjust_saturation(py, *factor)?,
                3 => current.adjust_hue(py, *factor)?,
                _ => current,
            };
        }

        Ok(current)
    }

    fn __repr__(&self) -> String {
        format!(
            "ColorJitter(brightness={:?}, contrast={:?}, saturation={:?}, hue={:?})",
            self.brightness, self.contrast, self.saturation, self.hue
        )
    }
}

/// Randomly flip image horizontally with probability p.
#[pyclass(name = "RandomHorizontalFlip")]
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
            Ok(PyImageApi::wrap(py, img.data(py), Some(img.mode().to_string())))
        }
    }

    fn __repr__(&self) -> String {
        format!("RandomHorizontalFlip(p={})", self.p)
    }
}

/// Randomly flip image vertically with probability p.
#[pyclass(name = "RandomVerticalFlip")]
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
            Ok(PyImageApi::wrap(py, img.data(py), Some(img.mode().to_string())))
        }
    }

    fn __repr__(&self) -> String {
        format!("RandomVerticalFlip(p={})", self.p)
    }
}

/// Randomly crop image to given size.
#[pyclass(name = "RandomCrop")]
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
#[pyclass(name = "RandomRotation")]
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
#[pyclass(name = "Compose")]
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
