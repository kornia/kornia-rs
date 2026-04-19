use numpy::{PyArray, PyArrayMethods};
use pyo3::prelude::*;
use pyo3::types::PyDict;
use rand::prelude::*;
use rand::rngs::StdRng;
use std::cell::RefCell;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Mutex;

use crate::image::{apply_brightness_sat, pyarray_data, PyImageApi, LUMINANCE_WEIGHTS};

// Fast path: unseeded mode checks a single Relaxed atomic load, skipping the
// Mutex entirely. The Mutex is only held briefly by `set_seed` to publish the
// seed, with a Release/Acquire pair making sure the SEEDED_RNG reseed below
// sees the new value.
static SEEDED_FLAG: AtomicBool = AtomicBool::new(false);
static GLOBAL_SEED: AtomicU64 = AtomicU64::new(0);
static SEED_WRITE_LOCK: Mutex<()> = Mutex::new(());

thread_local! {
    static SEEDED_RNG: RefCell<Option<StdRng>> = const { RefCell::new(None) };
    // The thread-local RNG is reseeded when this generation counter doesn't
    // match the global one. Avoids re-reading the global seed every call.
    static SEEDED_GEN: std::cell::Cell<u64> = const { std::cell::Cell::new(0) };
}

static SEED_GEN: AtomicU64 = AtomicU64::new(0);

#[inline]
fn with_rng<T>(f: impl FnOnce(&mut dyn RngCore) -> T) -> T {
    if SEEDED_FLAG.load(Ordering::Relaxed) {
        SEEDED_RNG.with(|cell| {
            let mut rng_opt = cell.borrow_mut();
            let cur_gen = SEED_GEN.load(Ordering::Acquire);
            let local_gen = SEEDED_GEN.with(|c| c.get());
            if rng_opt.is_none() || local_gen != cur_gen {
                let s = GLOBAL_SEED.load(Ordering::Relaxed);
                *rng_opt = Some(StdRng::seed_from_u64(s));
                SEEDED_GEN.with(|c| c.set(cur_gen));
            }
            f(rng_opt.as_mut().unwrap())
        })
    } else {
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
    let _g = SEED_WRITE_LOCK.lock().unwrap();
    match seed {
        Some(s) => {
            GLOBAL_SEED.store(s, Ordering::Relaxed);
            SEED_GEN.fetch_add(1, Ordering::Release);
            SEEDED_FLAG.store(true, Ordering::Release);
        }
        None => {
            SEEDED_FLAG.store(false, Ordering::Release);
            SEED_GEN.fetch_add(1, Ordering::Release);
        }
    }
    SEEDED_RNG.with(|cell| {
        *cell.borrow_mut() = seed.map(StdRng::seed_from_u64);
    });
    SEEDED_GEN.with(|c| c.set(SEED_GEN.load(Ordering::Relaxed)));
}

/// Extract a required key from a params dict.
///
/// Panics on missing key to match the previous `unwrap()` behavior at each call site.
fn dict_get<'py, T>(d: &Bound<'py, PyDict>, key: &str) -> PyResult<T>
where
    T: for<'a> FromPyObject<'a, 'py, Error = PyErr>,
{
    d.get_item(key)?.unwrap().extract::<T>()
}

/// Build a one-key `dict` (the common `last_params` shape).
fn dict_single<T: for<'py> pyo3::IntoPyObject<'py>>(
    py: Python<'_>,
    key: &str,
    value: T,
) -> PyResult<Py<PyDict>> {
    let d = PyDict::new(py);
    d.set_item(key, value)?;
    Ok(d.unbind())
}

fn check_input(value: f64, center: f64, bound: Option<(f64, f64)>) -> PyResult<(f64, f64)> {
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

// Fused ColorJitter kernel: linear ops (brightness/contrast) use LUT,
// nonlinear ops (saturation, hue) are per-pixel. Consecutive linear ops
// are collapsed into a single LUT pass.

const LW: [f32; 3] = [
    LUMINANCE_WEIGHTS[0] as f32,
    LUMINANCE_WEIGHTS[1] as f32,
    LUMINANCE_WEIGHTS[2] as f32,
];

#[inline]
fn build_linear_lut(b_off: f32, contrast: f32, mean: f32) -> [u8; 256] {
    let mut lut = [0u8; 256];
    for (v, slot) in lut.iter_mut().enumerate() {
        let f = mean + (v as f32 + b_off - mean) * contrast;
        *slot = f.clamp(0.0, 255.0) as u8;
    }
    lut
}

#[inline]
fn apply_lut(src: &[u8], dst: &mut [u8], lut: &[u8; 256]) {
    for (d, &s) in dst.iter_mut().zip(src.iter()) {
        *d = lut[s as usize];
    }
}

/// Mean of all bytes as f32, used as contrast pivot.
#[inline]
fn byte_mean(buf: &[u8]) -> f32 {
    let isum: u64 = buf.iter().map(|&v| v as u64).sum();
    isum as f32 / buf.len() as f32
}

/// Saturation adjustment (3ch), src → dst. Single fused pass.
///
/// For each pixel: `out_c = sat * src_c + (1 - sat) * gray` where
/// `gray = 0.299R + 0.587G + 0.114B`. `src` and `dst` may alias (same
/// buffer) since the kernel reads all three channels at a pixel before
/// writing. On aarch64 this runs a NEON Q8 fixed-point kernel; the
/// scalar fallback loops per pixel.
#[inline]
fn apply_saturation(src: &[u8], dst: &mut [u8], npixels: usize, saturation: f32) {
    #[cfg(target_arch = "aarch64")]
    unsafe {
        apply_saturation_neon(src, dst, npixels, saturation);
    }
    #[cfg(not(target_arch = "aarch64"))]
    apply_saturation_scalar(src, dst, npixels, saturation);
}

#[cfg(not(target_arch = "aarch64"))]
#[inline]
fn apply_saturation_scalar(src: &[u8], dst: &mut [u8], npixels: usize, saturation: f32) {
    let inv_sat = 1.0 - saturation;
    for i in 0..npixels {
        let base = i * 3;
        unsafe {
            let r = *src.get_unchecked(base) as f32;
            let g = *src.get_unchecked(base + 1) as f32;
            let b = *src.get_unchecked(base + 2) as f32;
            let gray = r * LW[0] + g * LW[1] + b * LW[2];
            let gw = gray * inv_sat;
            *dst.get_unchecked_mut(base) = (r * saturation + gw).clamp(0.0, 255.0) as u8;
            *dst.get_unchecked_mut(base + 1) = (g * saturation + gw).clamp(0.0, 255.0) as u8;
            *dst.get_unchecked_mut(base + 2) = (b * saturation + gw).clamp(0.0, 255.0) as u8;
        }
    }
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn apply_saturation_neon(src: &[u8], dst: &mut [u8], npixels: usize, saturation: f32) {
    use std::arch::aarch64::*;

    // Q8 fixed-point with i32 intermediates. `sat` is sampled in roughly
    // `[0, 2]` so `sat_q8 ∈ [0, 512]` fits in i16 (and the `delta * sat`
    // product fits in i32). i32 path is shorter than the f32 path — no
    // u32→f32→u32 conversion chains, and only one right-shift narrow to
    // finish. On A78AE this is ~4× faster than the f32 version.
    let sat_q8_s16 = vdupq_n_s16((saturation * 256.0).round() as i16);
    // Gray weights for u8 path: Y = (77R + 150G + 29B) >> 8 (same as
    // `rgb_to_gray_u8`). Max widened sum = 256 * 255 = 65280, fits in u16.
    let w_r = vdup_n_u8(77);
    let w_g = vdup_n_u8(150);
    let w_b = vdup_n_u8(29);

    let sptr = src.as_ptr();
    let dptr = dst.as_mut_ptr();
    let bulk16 = npixels & !15;
    let mut i = 0usize;
    while i < bulk16 {
        let ps = sptr.add(i * 3);
        let pd = dptr.add(i * 3);
        let rgb = vld3q_u8(ps);

        // gray u8x16 via Q8 weighted sum (77R + 150G + 29B) >> 8.
        let gray_lo = vmlal_u8(
            vmlal_u8(vmull_u8(vget_low_u8(rgb.0), w_r), vget_low_u8(rgb.1), w_g),
            vget_low_u8(rgb.2),
            w_b,
        );
        let gray_hi = vmlal_u8(
            vmlal_u8(vmull_u8(vget_high_u8(rgb.0), w_r), vget_high_u8(rgb.1), w_g),
            vget_high_u8(rgb.2),
            w_b,
        );
        let gray_u8 = vcombine_u8(vshrn_n_u16(gray_lo, 8), vshrn_n_u16(gray_hi, 8));
        let gray_s16_lo = vreinterpretq_s16_u16(vmovl_u8(vget_low_u8(gray_u8)));
        let gray_s16_hi = vreinterpretq_s16_u16(vmovl_u8(vget_high_u8(gray_u8)));

        // Per channel: out = gray + (sat_q8 * (src - gray) + rounding) >> 8,
        // clamped to [0, 255]. Implemented in i16/i32 via vmull_s16 + vrshrn.
        #[inline(always)]
        unsafe fn blend_channel(
            src_u8: uint8x16_t,
            gray_s16_lo: int16x8_t,
            gray_s16_hi: int16x8_t,
            sat_q8_s16: int16x8_t,
        ) -> uint8x16_t {
            let src_s16_lo = vreinterpretq_s16_u16(vmovl_u8(vget_low_u8(src_u8)));
            let src_s16_hi = vreinterpretq_s16_u16(vmovl_u8(vget_high_u8(src_u8)));
            let delta_lo = vsubq_s16(src_s16_lo, gray_s16_lo);
            let delta_hi = vsubq_s16(src_s16_hi, gray_s16_hi);
            // i16 * i16 → i32 widening mul, then rounding right shift 8 back to i16.
            let mul_lo_lo = vmull_s16(vget_low_s16(delta_lo), vget_low_s16(sat_q8_s16));
            let mul_lo_hi = vmull_s16(vget_high_s16(delta_lo), vget_high_s16(sat_q8_s16));
            let mul_hi_lo = vmull_s16(vget_low_s16(delta_hi), vget_low_s16(sat_q8_s16));
            let mul_hi_hi = vmull_s16(vget_high_s16(delta_hi), vget_high_s16(sat_q8_s16));
            let scaled_lo = vcombine_s16(vrshrn_n_s32(mul_lo_lo, 8), vrshrn_n_s32(mul_lo_hi, 8));
            let scaled_hi = vcombine_s16(vrshrn_n_s32(mul_hi_lo, 8), vrshrn_n_s32(mul_hi_hi, 8));
            let out_lo = vaddq_s16(gray_s16_lo, scaled_lo);
            let out_hi = vaddq_s16(gray_s16_hi, scaled_hi);
            // Saturating narrow to u8 clamps negative → 0 and > 255 → 255.
            vcombine_u8(vqmovun_s16(out_lo), vqmovun_s16(out_hi))
        }

        let out_r = blend_channel(rgb.0, gray_s16_lo, gray_s16_hi, sat_q8_s16);
        let out_g = blend_channel(rgb.1, gray_s16_lo, gray_s16_hi, sat_q8_s16);
        let out_b = blend_channel(rgb.2, gray_s16_lo, gray_s16_hi, sat_q8_s16);

        vst3q_u8(pd, uint8x16x3_t(out_r, out_g, out_b));

        i += 16;
    }

    // Scalar tail
    let inv_sat_s = 1.0 - saturation;
    while i < npixels {
        let base = i * 3;
        let r = *src.get_unchecked(base) as f32;
        let g = *src.get_unchecked(base + 1) as f32;
        let b = *src.get_unchecked(base + 2) as f32;
        let gray = r * LW[0] + g * LW[1] + b * LW[2];
        let gw = gray * inv_sat_s;
        *dst.get_unchecked_mut(base) = (r * saturation + gw).clamp(0.0, 255.0) as u8;
        *dst.get_unchecked_mut(base + 1) = (g * saturation + gw).clamp(0.0, 255.0) as u8;
        *dst.get_unchecked_mut(base + 2) = (b * saturation + gw).clamp(0.0, 255.0) as u8;
        i += 1;
    }
}

/// Hue rotation (3ch), src → dst. Rodrigues matrix around (1,1,1) — branchless.
///
/// `src` and `dst` may alias; the kernel reads all 3 channels before writing.
#[inline]
fn apply_hue(src: &[u8], dst: &mut [u8], npixels: usize, hue: f32) {
    let angle = hue * std::f32::consts::TAU;
    let cos_a = angle.cos();
    let sin_a = angle.sin();

    let one_third = 1.0 / 3.0;
    let sqrt_third = (1.0_f32 / 3.0).sqrt();

    let a = cos_a + one_third * (1.0 - cos_a);
    let b = one_third * (1.0 - cos_a) - sqrt_third * sin_a;
    let c = one_third * (1.0 - cos_a) + sqrt_third * sin_a;

    for i in 0..npixels {
        let base = i * 3;
        unsafe {
            let r = *src.get_unchecked(base) as f32;
            let g = *src.get_unchecked(base + 1) as f32;
            let b_val = *src.get_unchecked(base + 2) as f32;

            *dst.get_unchecked_mut(base) = (a * r + b * g + c * b_val).clamp(0.0, 255.0) as u8;
            *dst.get_unchecked_mut(base + 1) = (c * r + a * g + b * b_val).clamp(0.0, 255.0) as u8;
            *dst.get_unchecked_mut(base + 2) = (b * r + c * g + a * b_val).clamp(0.0, 255.0) as u8;
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn fused_color_jitter(
    src: &[u8],
    dst: &mut [u8],
    npixels: usize,
    c: usize,
    brightness: f32,
    contrast: f32,
    saturation: f32,
    hue: f32,
    order: &[(u8, f32)],
) {
    let has_brightness = brightness != 0.0;
    let has_contrast = contrast != 1.0;
    let has_saturation = saturation != 1.0 && c == 3;
    let has_hue = hue != 0.0 && c == 3;
    let has_linear = has_brightness || has_contrast;

    if !has_brightness && !has_contrast && !has_saturation && !has_hue {
        dst.copy_from_slice(src);
        return;
    }

    let mean: f32 = if has_contrast { byte_mean(src) } else { 0.0 };

    // Fast path: only linear ops — single pass from src to dst.
    if has_linear && !has_saturation && !has_hue {
        if has_brightness && !has_contrast {
            apply_brightness_sat(src, dst, brightness * 255.0);
        } else {
            let b_off = if has_brightness {
                brightness * 255.0
            } else {
                0.0
            };
            let c_f = if has_contrast { contrast } else { 1.0 };
            let lut = build_linear_lut(b_off, c_f, mean);
            apply_lut(src, dst, &lut);
        }
        return;
    }

    // Mixed path: avoid `dst.copy_from_slice(src)` — on a fresh PyArray the
    // copy page-faults across every 4KB dst page (~3ms at 1080p) without
    // producing useful output. Instead, the first op reads from src and writes
    // to dst directly (same page-fault cost, now productive work); later ops
    // run in-place on dst via the aliasing `slice::from_raw_parts(dst.as_ptr())`
    // trick, which is safe because each kernel reads a pixel before writing it.
    let mut pending_b: f32 = 0.0;
    let mut pending_c: f32 = 1.0;
    let mut pending_mean = mean;
    let mut has_pending_linear = false;
    let mut dst_init = false;

    for &(op, _) in order {
        match op {
            0 if has_brightness => {
                pending_b += brightness * 255.0;
                has_pending_linear = true;
            }
            1 if has_contrast => {
                pending_b *= contrast;
                pending_c *= contrast;
                has_pending_linear = true;
            }
            2 if has_saturation => {
                if has_pending_linear {
                    let input = current_src(src, dst, dst_init);
                    flush_linear(input, dst, pending_b, pending_c, pending_mean);
                    pending_b = 0.0;
                    pending_c = 1.0;
                    has_pending_linear = false;
                    dst_init = true;
                    if has_contrast {
                        pending_mean = byte_mean(dst);
                    }
                }
                let input = current_src(src, dst, dst_init);
                apply_saturation(input, dst, npixels, saturation);
                dst_init = true;
            }
            3 if has_hue => {
                if has_pending_linear {
                    let input = current_src(src, dst, dst_init);
                    flush_linear(input, dst, pending_b, pending_c, pending_mean);
                    pending_b = 0.0;
                    pending_c = 1.0;
                    has_pending_linear = false;
                    dst_init = true;
                    if has_contrast {
                        pending_mean = byte_mean(dst);
                    }
                }
                let input = current_src(src, dst, dst_init);
                apply_hue(input, dst, npixels, hue);
                dst_init = true;
            }
            _ => {}
        }
    }

    if has_pending_linear {
        let input = current_src(src, dst, dst_init);
        flush_linear(input, dst, pending_b, pending_c, pending_mean);
        dst_init = true;
    }

    // Mixed path requires has_saturation || has_hue, so dst_init is always true
    // here unless the sampled sat/hue collapsed to no-ops (e.g. 1.0 and 0.0
    // slipped past the caller filter). In that degenerate case we still need
    // dst to contain src.
    if !dst_init {
        dst.copy_from_slice(src);
    }
}

/// Return the correct input slice for the next op: the real `src` if `dst` has
/// not yet been written, otherwise a dst-aliased slice so the op runs in-place.
///
/// The returned slice has lifetime `'a` (tied to `src`), not to `dst`, so the
/// caller is free to pass `dst` mutably to the op. Safety: this is only sound
/// if each op reads a pixel before writing it, which holds for the LUT,
/// brightness, saturation, and hue kernels in this module.
#[inline]
fn current_src<'a>(src: &'a [u8], dst: &[u8], dst_init: bool) -> &'a [u8] {
    if dst_init {
        unsafe { std::slice::from_raw_parts(dst.as_ptr(), dst.len()) }
    } else {
        src
    }
}

/// Apply accumulated linear (brightness+contrast) transform, src → dst.
/// `src` and `dst` may alias (in-place operation).
#[inline]
fn flush_linear(src: &[u8], dst: &mut [u8], b: f32, c: f32, m: f32) {
    if c == 1.0 {
        apply_brightness_sat(src, dst, b);
    } else {
        let lut = build_linear_lut(b, c, m);
        apply_lut(src, dst, &lut);
    }
}

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
        let (b, c, s, h, order) = match params {
            Some(p) => Self::dict_to_params(p)?,
            None => self.sample_params(),
        };

        self.last_params = Some(Self::params_to_dict(py, b, c, s, h, &order)?);

        let arr = img.data(py);
        let bound = arr.bind(py);
        let (src, height, width, channels) = pyarray_data(bound);
        let npixels = height * width;
        let order_f32: Vec<(u8, f32)> = order.iter().map(|&(op, v)| (op, v as f32)).collect();

        // Allocate the output PyArray; write into it directly to avoid a copy.
        let out_arr = unsafe { PyArray::<u8, _>::new(py, [height, width, channels], false) };
        let dst = unsafe { std::slice::from_raw_parts_mut(out_arr.data(), src.len()) };

        py.detach(|| {
            fused_color_jitter(
                src, dst, npixels, channels, b as f32, c as f32, s as f32, h as f32, &order_f32,
            );
        });

        Ok(PyImageApi::wrap(
            py,
            out_arr.unbind(),
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

/// Tuple of (brightness, contrast, saturation, hue, op_order) sampled per call.
type JitterParams = (f64, f64, f64, f64, Vec<(u8, f64)>);

impl PyColorJitter {
    fn sample_params(&self) -> JitterParams {
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

    fn dict_to_params(d: &Bound<'_, PyDict>) -> PyResult<JitterParams> {
        let b: f64 = dict_get(d, "brightness")?;
        let c: f64 = dict_get(d, "contrast")?;
        let s: f64 = dict_get(d, "saturation")?;
        let h: f64 = dict_get(d, "hue")?;
        let order_list: Vec<u8> = dict_get(d, "order")?;
        let factors = [b, c, s, h];
        let order: Vec<(u8, f64)> = order_list
            .iter()
            .map(|&op| (op, factors[op as usize]))
            .collect();
        Ok((b, c, s, h, order))
    }
}

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
        dict_single(py, "flip", flip)
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
        let flip = match params {
            Some(p) => dict_get::<bool>(p, "flip")?,
            None => with_rng(|rng| rng.random::<f64>() < self.p),
        };

        self.last_params = Some(dict_single(py, "flip", flip)?);

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
        dict_single(py, "flip", flip)
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
        let flip = match params {
            Some(p) => dict_get::<bool>(p, "flip")?,
            None => with_rng(|rng| rng.random::<f64>() < self.p),
        };

        self.last_params = Some(dict_single(py, "flip", flip)?);

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

/// Randomly crop image to given size.
///
/// ``sample(img)`` returns ``{"x": int, "y": int}``.
/// ``last_params`` stores the last crop position.
#[pyclass(name = "RandomCrop")]
pub struct PyRandomCrop {
    height: usize,
    width: usize,
    last_xy: Option<(usize, usize)>,
}

#[pymethods]
impl PyRandomCrop {
    #[new]
    fn new(size: (usize, usize)) -> Self {
        Self {
            height: size.0,
            width: size.1,
            last_xy: None,
        }
    }

    /// Sample crop position. Requires the image to determine valid range.
    fn sample(&self, py: Python<'_>, img: PyRef<'_, PyImageApi>) -> PyResult<Py<PyDict>> {
        let (x, y) = self.sample_pos(img.width(py), img.height(py))?;
        Self::xy_dict(py, x, y)
    }

    /// The parameters used in the last ``__call__``. None before first call.
    #[getter]
    fn last_params(&self, py: Python<'_>) -> PyResult<Option<Py<PyDict>>> {
        self.last_xy
            .map(|(x, y)| Self::xy_dict(py, x, y))
            .transpose()
    }

    #[pyo3(signature = (img, params=None))]
    fn __call__(
        &mut self,
        py: Python<'_>,
        img: PyRef<'_, PyImageApi>,
        params: Option<&Bound<'_, PyDict>>,
    ) -> PyResult<PyImageApi> {
        let (x, y) = match params {
            Some(p) => (dict_get::<usize>(p, "x")?, dict_get::<usize>(p, "y")?),
            None => self.sample_pos(img.width(py), img.height(py))?,
        };

        self.last_xy = Some((x, y));

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

    fn xy_dict(py: Python<'_>, x: usize, y: usize) -> PyResult<Py<PyDict>> {
        let d = PyDict::new(py);
        d.set_item("x", x)?;
        d.set_item("y", y)?;
        Ok(d.unbind())
    }
}

/// Randomly rotate image within degree range.
///
/// ``sample()`` returns ``{"angle": float}``.
/// ``last_params`` stores the last rotation angle.
#[pyclass(name = "RandomRotation")]
pub struct PyRandomRotation {
    degrees: (f64, f64),
    last_angle: Option<f64>,
}

#[pymethods]
impl PyRandomRotation {
    #[new]
    fn new(degrees: f64) -> Self {
        Self {
            degrees: (-degrees, degrees),
            last_angle: None,
        }
    }

    fn sample(&self, py: Python<'_>) -> PyResult<Py<PyDict>> {
        let angle: f64 = with_rng(|rng| rng.random_range(self.degrees.0..=self.degrees.1));
        dict_single(py, "angle", angle)
    }

    /// The parameters used in the last ``__call__``. None before first call.
    #[getter]
    fn last_params(&self, py: Python<'_>) -> PyResult<Option<Py<PyDict>>> {
        self.last_angle
            .map(|a| dict_single(py, "angle", a))
            .transpose()
    }

    #[pyo3(signature = (img, params=None))]
    fn __call__(
        &mut self,
        py: Python<'_>,
        img: PyRef<'_, PyImageApi>,
        params: Option<&Bound<'_, PyDict>>,
    ) -> PyResult<PyImageApi> {
        let angle = match params {
            Some(p) => dict_get::<f64>(p, "angle")?,
            None => with_rng(|rng| rng.random_range(self.degrees.0..=self.degrees.1)),
        };

        self.last_angle = Some(angle);

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
