use pyo3::prelude::*;
use pyo3::types::PyDict;
use rand::prelude::*;
use rand::rngs::StdRng;
use std::cell::RefCell;
use std::sync::Mutex;

use crate::image::{pyarray_data, vec_to_pyarray, PyImageApi, LUMINANCE_WEIGHTS};

static GLOBAL_SEED: Mutex<Option<u64>> = Mutex::new(None);

thread_local! {
    static SEEDED_RNG: RefCell<Option<StdRng>> = const { RefCell::new(None) };
}

fn with_rng<T>(f: impl FnOnce(&mut dyn RngCore) -> T) -> T {
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
    SEEDED_RNG.with(|cell| {
        *cell.borrow_mut() = seed.map(StdRng::seed_from_u64);
    });
}

fn check_input(
    value: f64,
    center: f64,
    bound: Option<(f64, f64)>,
) -> PyResult<(f64, f64)> {
    if value < 0.0 {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
            "value must be non-negative, got {}",
            value
        )));
    }
    let range = if center == 0.0 {
        (-value, value)
    } else {
        (f64::max(0.0, center - value), center + value)
    };
    if let Some((lo, hi)) = bound {
        if range.0 < lo || range.1 > hi {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                "values should be between ({}, {})",
                lo, hi
            )));
        }
    }
    Ok(range)
}

// ---------------------------------------------------------------------------
// Fused ColorJitter kernel
// ---------------------------------------------------------------------------
//
// Fuses brightness, contrast, saturation, and hue into at most 2 passes:
//   Pass 1 (if contrast active): compute pixel mean
//   Pass 2: apply all ops per-pixel in one sweep
//
// Brightness + contrast are linear:  out = pixel * c_factor + (b_offset*255 * c_factor + mean * (1 - c_factor))
// Saturation + hue share a single RGB→HSV→RGB round-trip.

fn fused_color_jitter(
    src: &[u8],
    npixels: usize,
    c: usize,
    brightness: f64,
    contrast: f64,
    saturation: f64,
    hue: f64,
    order: &[(u8, f64)],
) -> Vec<u8> {
    let has_brightness = brightness != 0.0;
    let has_contrast = contrast != 1.0;
    let has_saturation = saturation != 1.0;
    let has_hue = hue != 0.0;

    // Fast path: no ops active
    if !has_brightness && !has_contrast && !has_saturation && !has_hue {
        return src.to_vec();
    }

    // Only need mean if contrast is active
    let mean = if has_contrast {
        src.iter().map(|&v| v as f64).sum::<f64>() / src.len() as f64
    } else {
        0.0
    };

    let mut out = vec![0u8; npixels * c];

    // For non-3ch images, only brightness and contrast apply
    if c != 3 {
        let b_offset = brightness * 255.0;
        for i in 0..src.len() {
            let mut v = src[i] as f64 + b_offset;
            if has_contrast {
                v = mean + (v - mean) * contrast;
            }
            out[i] = v.clamp(0.0, 255.0) as u8;
        }
        return out;
    }

    // 3-channel: apply all ops per-pixel following the sampled order.
    // We process each pixel through the ordered ops sequentially.
    for i in 0..npixels {
        let mut r = src[i * 3] as f64;
        let mut g = src[i * 3 + 1] as f64;
        let mut b = src[i * 3 + 2] as f64;

        for &(op, _) in order {
            match op {
                0 if has_brightness => {
                    let offset = brightness * 255.0;
                    r += offset;
                    g += offset;
                    b += offset;
                    r = r.clamp(0.0, 255.0);
                    g = g.clamp(0.0, 255.0);
                    b = b.clamp(0.0, 255.0);
                }
                1 if has_contrast => {
                    r = mean + (r - mean) * contrast;
                    g = mean + (g - mean) * contrast;
                    b = mean + (b - mean) * contrast;
                    r = r.clamp(0.0, 255.0);
                    g = g.clamp(0.0, 255.0);
                    b = b.clamp(0.0, 255.0);
                }
                2 if has_saturation => {
                    let gray = r * LUMINANCE_WEIGHTS[0]
                        + g * LUMINANCE_WEIGHTS[1]
                        + b * LUMINANCE_WEIGHTS[2];
                    r = (gray + (r - gray) * saturation).clamp(0.0, 255.0);
                    g = (gray + (g - gray) * saturation).clamp(0.0, 255.0);
                    b = (gray + (b - gray) * saturation).clamp(0.0, 255.0);
                }
                3 if has_hue => {
                    let rn = r / 255.0;
                    let gn = g / 255.0;
                    let bn = b / 255.0;

                    let max = rn.max(gn).max(bn);
                    let min = rn.min(gn).min(bn);
                    let diff = max - min;

                    let mut h = if diff < 1e-10 {
                        0.0
                    } else if (max - rn).abs() < 1e-10 {
                        ((gn - bn) / diff).rem_euclid(6.0)
                    } else if (max - gn).abs() < 1e-10 {
                        (bn - rn) / diff + 2.0
                    } else {
                        (rn - gn) / diff + 4.0
                    };
                    h /= 6.0;

                    let s = if max < 1e-10 { 0.0 } else { diff / max };
                    let v = max;

                    h = (h + hue).rem_euclid(1.0);

                    let h6 = h * 6.0;
                    let hi = h6.floor() as i32 % 6;
                    let f = h6 - h6.floor();
                    let p = v * (1.0 - s);
                    let q = v * (1.0 - s * f);
                    let t = v * (1.0 - s * (1.0 - f));

                    let (ro, go, bo) = match hi {
                        0 => (v, t, p),
                        1 => (q, v, p),
                        2 => (p, v, t),
                        3 => (p, q, v),
                        4 => (t, p, v),
                        _ => (v, p, q),
                    };

                    r = (ro * 255.0).clamp(0.0, 255.0);
                    g = (go * 255.0).clamp(0.0, 255.0);
                    b = (bo * 255.0).clamp(0.0, 255.0);
                }
                _ => {}
            }
        }

        out[i * 3] = r as u8;
        out[i * 3 + 1] = g as u8;
        out[i * 3 + 2] = b as u8;
    }
    out
}

// ---------------------------------------------------------------------------
// ColorJitter
// ---------------------------------------------------------------------------

/// Randomly change brightness, contrast, saturation, and hue.
///
/// All four operations are fused into a single pass over the pixel data,
/// avoiding intermediate allocations.
///
/// Use ``sample()`` to pre-generate parameters (e.g. to apply the same
/// jitter to an image and its mask), and ``last_params`` to inspect
/// what was applied.
#[pyclass(name = "ColorJitter")]
pub struct PyColorJitter {
    brightness: (f64, f64),
    contrast: (f64, f64),
    saturation: (f64, f64),
    hue: (f64, f64),
    last_params: Option<Py<PyDict>>,
}

#[pymethods]
impl PyColorJitter {
    #[new]
    #[pyo3(signature = (brightness=0.0, contrast=0.0, saturation=0.0, hue=0.0))]
    fn new(brightness: f64, contrast: f64, saturation: f64, hue: f64) -> PyResult<Self> {
        Ok(Self {
            brightness: check_input(brightness, 0.0, Some((-1.0, 1.0)))?,
            contrast: check_input(contrast, 1.0, None)?,
            saturation: check_input(saturation, 1.0, None)?,
            hue: check_input(hue, 0.0, Some((-0.5, 0.5)))?,
            last_params: None,
        })
    }

    /// Sample random parameters without applying them.
    ///
    /// Returns a dict with keys: brightness, contrast, saturation, hue, order.
    /// Pass the dict to ``__call__`` to apply the same jitter to multiple images.
    fn sample(&self, py: Python<'_>) -> PyResult<Py<PyDict>> {
        let (b, c, s, h, order) = self.sample_params();
        Self::params_to_dict(py, b, c, s, h, &order)
    }

    /// The parameters used in the last ``__call__``. None before first call.
    #[getter]
    fn last_params(&self, py: Python<'_>) -> Option<Py<PyDict>> {
        self.last_params.as_ref().map(|p| p.clone_ref(py))
    }

    #[pyo3(signature = (img, params=None))]
    fn __call__(
        &mut self,
        py: Python<'_>,
        img: PyRef<'_, PyImageApi>,
        params: Option<&Bound<'_, PyDict>>,
    ) -> PyResult<PyImageApi> {
        let (b, c, s, h, order) = if let Some(p) = params {
            Self::dict_to_params(p)?
        } else {
            self.sample_params()
        };

        self.last_params = Some(Self::params_to_dict(py, b, c, s, h, &order)?);

        let arr = img.data(py);
        let bound = arr.bind(py);
        let (src, height, width, channels) = pyarray_data(bound);
        let npixels = height * width;

        let out = fused_color_jitter(src, npixels, channels, b, c, s, h, &order);

        Ok(PyImageApi::wrap(
            py,
            vec_to_pyarray(py, out, height, width, channels),
            Some(img.mode().to_string()),
        ))
    }

    fn __repr__(&self) -> String {
        format!(
            "ColorJitter(brightness={:?}, contrast={:?}, saturation={:?}, hue={:?})",
            self.brightness, self.contrast, self.saturation, self.hue
        )
    }
}

impl PyColorJitter {
    fn sample_params(&self) -> (f64, f64, f64, f64, Vec<(u8, f64)>) {
        let mut ops: Vec<(u8, f64)> = Vec::with_capacity(4);
        let (mut b, mut c, mut s, mut h) = (0.0, 1.0, 1.0, 0.0);

        with_rng(|rng| {
            if self.brightness.0 != 0.0 || self.brightness.1 != 0.0 {
                b = rng.random_range(self.brightness.0..=self.brightness.1);
                ops.push((0, b));
            }
            if self.contrast.0 != 1.0 || self.contrast.1 != 1.0 {
                c = rng.random_range(self.contrast.0..=self.contrast.1);
                ops.push((1, c));
            }
            if self.saturation.0 != 1.0 || self.saturation.1 != 1.0 {
                s = rng.random_range(self.saturation.0..=self.saturation.1);
                ops.push((2, s));
            }
            if self.hue.0 != 0.0 || self.hue.1 != 0.0 {
                h = rng.random_range(self.hue.0..=self.hue.1);
                ops.push((3, h));
            }
            ops.shuffle(rng);
        });

        (b, c, s, h, ops)
    }

    fn params_to_dict(
        py: Python<'_>,
        b: f64,
        c: f64,
        s: f64,
        h: f64,
        order: &[(u8, f64)],
    ) -> PyResult<Py<PyDict>> {
        let d = PyDict::new(py);
        d.set_item("brightness", b)?;
        d.set_item("contrast", c)?;
        d.set_item("saturation", s)?;
        d.set_item("hue", h)?;
        let order_list: Vec<u8> = order.iter().map(|(op, _)| *op).collect();
        d.set_item("order", order_list)?;
        Ok(d.unbind())
    }

    fn dict_to_params(d: &Bound<'_, PyDict>) -> PyResult<(f64, f64, f64, f64, Vec<(u8, f64)>)> {
        let b: f64 = d.get_item("brightness")?.unwrap().extract()?;
        let c: f64 = d.get_item("contrast")?.unwrap().extract()?;
        let s: f64 = d.get_item("saturation")?.unwrap().extract()?;
        let h: f64 = d.get_item("hue")?.unwrap().extract()?;
        let order_list: Vec<u8> = d.get_item("order")?.unwrap().extract()?;
        let factors = [b, c, s, h];
        let order: Vec<(u8, f64)> = order_list.iter().map(|&op| (op, factors[op as usize])).collect();
        Ok((b, c, s, h, order))
    }
}

// ---------------------------------------------------------------------------
// RandomHorizontalFlip
// ---------------------------------------------------------------------------

/// Randomly flip image horizontally with probability p.
///
/// ``sample()`` returns ``{"flip": bool}``.
/// ``last_params`` stores the last decision.
#[pyclass(name = "RandomHorizontalFlip")]
pub struct PyRandomHorizontalFlip {
    p: f64,
    last_params: Option<Py<PyDict>>,
}

#[pymethods]
impl PyRandomHorizontalFlip {
    #[new]
    #[pyo3(signature = (p=0.5))]
    fn new(p: f64) -> Self {
        Self {
            p,
            last_params: None,
        }
    }

    fn sample(&self, py: Python<'_>) -> PyResult<Py<PyDict>> {
        let flip: bool = with_rng(|rng| rng.random::<f64>() < self.p);
        let d = PyDict::new(py);
        d.set_item("flip", flip)?;
        Ok(d.unbind())
    }

    #[getter]
    fn last_params(&self, py: Python<'_>) -> Option<Py<PyDict>> {
        self.last_params.as_ref().map(|p| p.clone_ref(py))
    }

    #[pyo3(signature = (img, params=None))]
    fn __call__(
        &mut self,
        py: Python<'_>,
        img: PyRef<'_, PyImageApi>,
        params: Option<&Bound<'_, PyDict>>,
    ) -> PyResult<PyImageApi> {
        let flip = if let Some(p) = params {
            p.get_item("flip")?.unwrap().extract::<bool>()?
        } else {
            with_rng(|rng| rng.random::<f64>() < self.p)
        };

        let d = PyDict::new(py);
        d.set_item("flip", flip)?;
        self.last_params = Some(d.unbind());

        if flip {
            img.flip_horizontal(py)
        } else {
            Ok(PyImageApi::wrap(
                py,
                img.data(py),
                Some(img.mode().to_string()),
            ))
        }
    }

    fn __repr__(&self) -> String {
        format!("RandomHorizontalFlip(p={})", self.p)
    }
}

// ---------------------------------------------------------------------------
// RandomVerticalFlip
// ---------------------------------------------------------------------------

/// Randomly flip image vertically with probability p.
///
/// ``sample()`` returns ``{"flip": bool}``.
/// ``last_params`` stores the last decision.
#[pyclass(name = "RandomVerticalFlip")]
pub struct PyRandomVerticalFlip {
    p: f64,
    last_params: Option<Py<PyDict>>,
}

#[pymethods]
impl PyRandomVerticalFlip {
    #[new]
    #[pyo3(signature = (p=0.5))]
    fn new(p: f64) -> Self {
        Self {
            p,
            last_params: None,
        }
    }

    fn sample(&self, py: Python<'_>) -> PyResult<Py<PyDict>> {
        let flip: bool = with_rng(|rng| rng.random::<f64>() < self.p);
        let d = PyDict::new(py);
        d.set_item("flip", flip)?;
        Ok(d.unbind())
    }

    #[getter]
    fn last_params(&self, py: Python<'_>) -> Option<Py<PyDict>> {
        self.last_params.as_ref().map(|p| p.clone_ref(py))
    }

    #[pyo3(signature = (img, params=None))]
    fn __call__(
        &mut self,
        py: Python<'_>,
        img: PyRef<'_, PyImageApi>,
        params: Option<&Bound<'_, PyDict>>,
    ) -> PyResult<PyImageApi> {
        let flip = if let Some(p) = params {
            p.get_item("flip")?.unwrap().extract::<bool>()?
        } else {
            with_rng(|rng| rng.random::<f64>() < self.p)
        };

        let d = PyDict::new(py);
        d.set_item("flip", flip)?;
        self.last_params = Some(d.unbind());

        if flip {
            img.flip_vertical(py)
        } else {
            Ok(PyImageApi::wrap(
                py,
                img.data(py),
                Some(img.mode().to_string()),
            ))
        }
    }

    fn __repr__(&self) -> String {
        format!("RandomVerticalFlip(p={})", self.p)
    }
}

// ---------------------------------------------------------------------------
// RandomCrop
// ---------------------------------------------------------------------------

/// Randomly crop image to given size.
///
/// ``sample(img)`` returns ``{"x": int, "y": int}``.
/// ``last_params`` stores the last crop position.
#[pyclass(name = "RandomCrop")]
pub struct PyRandomCrop {
    height: usize,
    width: usize,
    last_params: Option<Py<PyDict>>,
}

#[pymethods]
impl PyRandomCrop {
    #[new]
    fn new(size: (usize, usize)) -> Self {
        Self {
            height: size.0,
            width: size.1,
            last_params: None,
        }
    }

    /// Sample crop position. Requires the image to determine valid range.
    fn sample(
        &self,
        py: Python<'_>,
        img: PyRef<'_, PyImageApi>,
    ) -> PyResult<Py<PyDict>> {
        let (x, y) = self.sample_pos(img.width(py), img.height(py))?;
        let d = PyDict::new(py);
        d.set_item("x", x)?;
        d.set_item("y", y)?;
        Ok(d.unbind())
    }

    #[getter]
    fn last_params(&self, py: Python<'_>) -> Option<Py<PyDict>> {
        self.last_params.as_ref().map(|p| p.clone_ref(py))
    }

    #[pyo3(signature = (img, params=None))]
    fn __call__(
        &mut self,
        py: Python<'_>,
        img: PyRef<'_, PyImageApi>,
        params: Option<&Bound<'_, PyDict>>,
    ) -> PyResult<PyImageApi> {
        let (x, y) = if let Some(p) = params {
            (
                p.get_item("x")?.unwrap().extract::<usize>()?,
                p.get_item("y")?.unwrap().extract::<usize>()?,
            )
        } else {
            self.sample_pos(img.width(py), img.height(py))?
        };

        let d = PyDict::new(py);
        d.set_item("x", x)?;
        d.set_item("y", y)?;
        self.last_params = Some(d.unbind());

        img.crop(py, x, y, self.width, self.height)
    }

    fn __repr__(&self) -> String {
        format!("RandomCrop(size=({}, {}))", self.height, self.width)
    }
}

impl PyRandomCrop {
    fn sample_pos(&self, img_w: usize, img_h: usize) -> PyResult<(usize, usize)> {
        if img_w < self.width || img_h < self.height {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                "Image ({}x{}) is smaller than crop size ({}x{})",
                img_w, img_h, self.width, self.height
            )));
        }
        Ok(with_rng(|rng| {
            let x = rng.random_range(0..=(img_w - self.width));
            let y = rng.random_range(0..=(img_h - self.height));
            (x, y)
        }))
    }
}

// ---------------------------------------------------------------------------
// RandomRotation
// ---------------------------------------------------------------------------

/// Randomly rotate image within degree range.
///
/// ``sample()`` returns ``{"angle": float}``.
/// ``last_params`` stores the last rotation angle.
#[pyclass(name = "RandomRotation")]
pub struct PyRandomRotation {
    degrees: (f64, f64),
    last_params: Option<Py<PyDict>>,
}

#[pymethods]
impl PyRandomRotation {
    #[new]
    fn new(degrees: f64) -> Self {
        Self {
            degrees: (-degrees, degrees),
            last_params: None,
        }
    }

    fn sample(&self, py: Python<'_>) -> PyResult<Py<PyDict>> {
        let angle: f64 = with_rng(|rng| rng.random_range(self.degrees.0..=self.degrees.1));
        let d = PyDict::new(py);
        d.set_item("angle", angle)?;
        Ok(d.unbind())
    }

    #[getter]
    fn last_params(&self, py: Python<'_>) -> Option<Py<PyDict>> {
        self.last_params.as_ref().map(|p| p.clone_ref(py))
    }

    #[pyo3(signature = (img, params=None))]
    fn __call__(
        &mut self,
        py: Python<'_>,
        img: PyRef<'_, PyImageApi>,
        params: Option<&Bound<'_, PyDict>>,
    ) -> PyResult<PyImageApi> {
        let angle = if let Some(p) = params {
            p.get_item("angle")?.unwrap().extract::<f64>()?
        } else {
            with_rng(|rng| rng.random_range(self.degrees.0..=self.degrees.1))
        };

        let d = PyDict::new(py);
        d.set_item("angle", angle)?;
        self.last_params = Some(d.unbind());

        img.rotate(py, angle)
    }

    fn __repr__(&self) -> String {
        format!("RandomRotation(degrees={:?})", self.degrees)
    }
}

// ---------------------------------------------------------------------------
// Compose
// ---------------------------------------------------------------------------

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
