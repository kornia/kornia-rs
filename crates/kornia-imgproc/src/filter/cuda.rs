//! Device adapters for the separable-filter ops (residency-dispatch arms).
//!
//! The 1D tap tables are built by the SAME host functions the CPU paths use
//! (`filter/kernels.rs`, `quantize_kernel_256`) and uploaded once per
//! parameter set through a small device cache (Jetson pageable H2D has a
//! ~250 µs latency tail; the cache synchronizes its uploading stream before
//! an entry becomes visible, so cross-stream hits always read completed
//! tables, and evicts one entry at cap).
//!
//! Scratch buffers are per-call stream-ordered allocations (mempool-cheap;
//! not CUDA-graph-capturable — same documented limitation as the morphology
//! separable path; a Plan object is the follow-up for both).

use std::collections::HashMap;
use std::sync::{Arc, Mutex, OnceLock};

use cudarc::driver::{CudaSlice, CudaStream};
use kornia_image::{Image, ImageError};

use crate::cuda::dispatch::{device_slices, dims_u32, untyped_device_err};
use crate::cuda::filter::{
    launch_binomial3_u8, launch_gradient_magnitude_f32, launch_separable_blur_u8q8,
    launch_separable_filter_f32,
};

// ── tap-table device cache ───────────────────────────────────────────────────

#[derive(Clone, PartialEq, Eq, Hash)]
struct TapKey {
    dev: usize,
    /// Tap values as raw bit patterns (f32 taps) or widened bytes (u8 taps),
    /// so one map serves both dtypes without float-Eq issues.
    bits: Vec<u32>,
    is_u8: bool,
}

enum TapTable {
    F32(Arc<CudaSlice<f32>>),
    U8(Arc<CudaSlice<u8>>),
}

type TapCache = Mutex<HashMap<TapKey, TapTable>>;
static TAP_CACHE: OnceLock<TapCache> = OnceLock::new();
const TAP_CACHE_CAP: usize = 128;

fn err_cuda(e: impl std::fmt::Display) -> ImageError {
    ImageError::Cuda(e.to_string())
}

fn insert_tap(key: TapKey, built: TapTable, stream: &Arc<CudaStream>) -> Result<(), ImageError> {
    // Uploads are async on THIS stream but the cache is shared across
    // streams; synchronize once before the entry becomes visible.
    stream.synchronize().map_err(err_cuda)?;
    let mut map = TAP_CACHE
        .get_or_init(Default::default)
        .lock()
        .expect("filter tap cache poisoned");
    if map.len() >= TAP_CACHE_CAP {
        if let Some(k) = map.keys().next().cloned() {
            map.remove(&k);
        }
    }
    map.entry(key).or_insert(built);
    Ok(())
}

fn cached_taps_f32(
    stream: &Arc<CudaStream>,
    taps: &[f32],
) -> Result<Arc<CudaSlice<f32>>, ImageError> {
    let key = TapKey {
        dev: stream.context().ordinal(),
        bits: taps.iter().map(|t| t.to_bits()).collect(),
        is_u8: false,
    };
    if let Some(TapTable::F32(hit)) = TAP_CACHE
        .get_or_init(Default::default)
        .lock()
        .expect("filter tap cache poisoned")
        .get(&key)
        .map(|t| match t {
            TapTable::F32(a) => TapTable::F32(a.clone()),
            TapTable::U8(a) => TapTable::U8(a.clone()),
        })
    {
        return Ok(hit);
    }
    let built = Arc::new(stream.clone_htod(taps).map_err(err_cuda)?);
    insert_tap(key, TapTable::F32(built.clone()), stream)?;
    Ok(built)
}

fn cached_taps_u8(stream: &Arc<CudaStream>, taps: &[u8]) -> Result<Arc<CudaSlice<u8>>, ImageError> {
    let key = TapKey {
        dev: stream.context().ordinal(),
        bits: taps.iter().map(|&t| t as u32).collect(),
        is_u8: true,
    };
    if let Some(TapTable::U8(hit)) = TAP_CACHE
        .get_or_init(Default::default)
        .lock()
        .expect("filter tap cache poisoned")
        .get(&key)
        .map(|t| match t {
            TapTable::F32(a) => TapTable::F32(a.clone()),
            TapTable::U8(a) => TapTable::U8(a.clone()),
        })
    {
        return Ok(hit);
    }
    let built = Arc::new(stream.clone_htod(taps).map_err(err_cuda)?);
    insert_tap(key, TapTable::U8(built.clone()), stream)?;
    Ok(built)
}

// ── adapters ─────────────────────────────────────────────────────────────────

/// Device twin of `separable_filter` for f32 images — bit-exact (skip-zero
/// border, sequential taps, `--fmad=false` kernels).
pub(super) fn separable_filter_f32_cuda<const C: usize>(
    src: &Image<f32, C>,
    dst: &mut Image<f32, C>,
    kernel_x: &[f32],
    kernel_y: &[f32],
    stream: &Arc<CudaStream>,
) -> Result<(), ImageError> {
    let (cols, rows) = dims_u32(src)?;
    let ctx = stream.context();
    let (s, d) = device_slices!(src, dst);
    let kx = cached_taps_f32(stream, kernel_x)?;
    let ky = cached_taps_f32(stream, kernel_y)?;
    let n = cols as usize * rows as usize * C;
    let mut scratch = unsafe { stream.alloc::<f32>(n) }.map_err(err_cuda)?;
    launch_separable_filter_f32(
        ctx,
        stream,
        s,
        d,
        &mut scratch,
        &kx,
        kernel_x.len() as u32,
        &ky,
        kernel_y.len() as u32,
        cols,
        rows,
        C as u32,
    )
    .map_err(err_cuda)
}

/// Device twin of the u8 Q8 striped blur (replicate borders, Q8 per pass).
pub(super) fn separable_blur_u8_cuda<const C: usize>(
    src: &Image<u8, C>,
    dst: &mut Image<u8, C>,
    ikx: &[u8],
    iky: &[u8],
    stream: &Arc<CudaStream>,
) -> Result<(), ImageError> {
    let (cols, rows) = dims_u32(src)?;
    let ctx = stream.context();
    let (s, d) = device_slices!(src, dst);
    let kx = cached_taps_u8(stream, ikx)?;
    let ky = cached_taps_u8(stream, iky)?;
    let n = cols as usize * rows as usize * C;
    let mut scratch = unsafe { stream.alloc::<u8>(n) }.map_err(err_cuda)?;
    launch_separable_blur_u8q8(
        ctx,
        stream,
        s,
        d,
        &mut scratch,
        &kx,
        ikx.len() as u32,
        &ky,
        iky.len() as u32,
        cols,
        rows,
        C as u32,
    )
    .map_err(err_cuda)
}

/// Device twin of the u8 3×3 binomial fast path (nested halving-adds).
pub(super) fn binomial3_u8_cuda<const C: usize>(
    src: &Image<u8, C>,
    dst: &mut Image<u8, C>,
    stream: &Arc<CudaStream>,
) -> Result<(), ImageError> {
    let (cols, rows) = dims_u32(src)?;
    let ctx = stream.context();
    let (s, d) = device_slices!(src, dst);
    let n = cols as usize * rows as usize * C;
    let mut scratch = unsafe { stream.alloc::<u8>(n) }.map_err(err_cuda)?;
    launch_binomial3_u8(ctx, stream, s, d, &mut scratch, cols, rows, C as u32).map_err(err_cuda)
}

/// Device twin of `sobel`/`scharr`: two separable passes into gx/gy scratch,
/// then the magnitude fold — every stage bit-exact with its CPU twin.
pub(super) fn gradient_magnitude_f32_cuda<const C: usize>(
    src: &Image<f32, C>,
    dst: &mut Image<f32, C>,
    kernel_a: &[f32],
    kernel_b: &[f32],
    stream: &Arc<CudaStream>,
) -> Result<(), ImageError> {
    let (cols, rows) = dims_u32(src)?;
    let ctx = stream.context();
    let (s, d) = device_slices!(src, dst);
    let ka = cached_taps_f32(stream, kernel_a)?;
    let kb = cached_taps_f32(stream, kernel_b)?;
    let n = cols as usize * rows as usize * C;
    let mut scratch = unsafe { stream.alloc::<f32>(n) }.map_err(err_cuda)?;
    let mut gx = unsafe { stream.alloc::<f32>(n) }.map_err(err_cuda)?;
    let mut gy = unsafe { stream.alloc::<f32>(n) }.map_err(err_cuda)?;
    // gx: kernel_a horizontal, kernel_b vertical; gy: swapped — the same
    // (kernel_x, kernel_y) pairing the CPU sobel/scharr use.
    launch_separable_filter_f32(
        ctx,
        stream,
        s,
        &mut gx,
        &mut scratch,
        &ka,
        kernel_a.len() as u32,
        &kb,
        kernel_b.len() as u32,
        cols,
        rows,
        C as u32,
    )
    .map_err(err_cuda)?;
    launch_separable_filter_f32(
        ctx,
        stream,
        s,
        &mut gy,
        &mut scratch,
        &kb,
        kernel_b.len() as u32,
        &ka,
        kernel_a.len() as u32,
        cols,
        rows,
        C as u32,
    )
    .map_err(err_cuda)?;
    launch_gradient_magnitude_f32(ctx, stream, &gx, &gy, d, n).map_err(err_cuda)
}

// ── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use crate::cuda::color::test_utils::{default_stream, pattern_f32, pattern_u8};
    use crate::filter::{box_blur, box_blur_u8, gaussian_blur, gaussian_blur_u8, scharr, sobel};
    use kornia_image::{Image, ImageSize};

    fn sized(w: usize, h: usize) -> ImageSize {
        ImageSize {
            width: w,
            height: h,
        }
    }

    /// Every f32 separable op must be bit-identical to its CPU twin
    /// (skip-zero border, sequential taps, fmad=false).
    #[test]
    fn f32_filters_device_equal_host_bitexact() {
        let stream = default_stream();
        let (w, h) = (67, 43);
        let src = Image::<f32, 3>::new(sized(w, h), pattern_f32(w * h * 3)).unwrap();
        let d_src = src.to_cuda(&stream).unwrap();

        type OpFn = fn(&Image<f32, 3>, &mut Image<f32, 3>) -> Result<(), kornia_image::ImageError>;
        let cases: &[(&str, OpFn)] = &[
            ("box_blur5x3", |s, d| box_blur(s, d, (5, 3))),
            ("gaussian5", |s, d| gaussian_blur(s, d, (5, 5), (1.5, 1.5))),
            ("sobel3", |s, d| sobel(s, d, 3)),
            ("scharr3", |s, d| scharr(s, d, 3)),
        ];
        for (name, op) in cases {
            let mut cpu = Image::<f32, 3>::from_size_val(sized(w, h), 0.0).unwrap();
            op(&src, &mut cpu).unwrap();
            let mut d_dst = Image::<f32, 3>::zeros_cuda(sized(w, h), &stream).unwrap();
            op(&d_src, &mut d_dst).unwrap();
            let back = d_dst.to_host_owned().unwrap();
            for (i, (a, b)) in back.as_slice().iter().zip(cpu.as_slice()).enumerate() {
                assert!(a.to_bits() == b.to_bits(), "{name} elem {i}: {a} vs {b}");
            }
        }
    }

    /// u8 blurs must be byte-exact vs the CPU paths for BOTH selector
    /// branches (binomial 3x3 and general Q8) and both u8 ops.
    #[test]
    fn u8_blurs_device_equal_host_byte_exact() {
        let stream = default_stream();
        for (w, h) in [(64usize, 48usize), (67, 43)] {
            let src = Image::<u8, 1>::new(sized(w, h), pattern_u8(w * h)).unwrap();
            let src3 = Image::<u8, 3>::new(sized(w, h), pattern_u8(w * h * 3)).unwrap();
            let d_src = src.to_cuda(&stream).unwrap();
            let d_src3 = src3.to_cuda(&stream).unwrap();

            // Binomial branch (k=3, sigma 1.0).
            let mut cpu = Image::<u8, 1>::from_size_val(sized(w, h), 0).unwrap();
            gaussian_blur_u8(&src, &mut cpu, (3, 3), (1.0, 1.0)).unwrap();
            let mut d_dst = Image::<u8, 1>::zeros_cuda(sized(w, h), &stream).unwrap();
            gaussian_blur_u8(&d_src, &mut d_dst, (3, 3), (1.0, 1.0)).unwrap();
            assert_eq!(
                d_dst.to_host_owned().unwrap().as_slice(),
                cpu.as_slice(),
                "binomial {w}x{h}"
            );

            // General Q8 branch (k=5).
            let mut cpu = Image::<u8, 3>::from_size_val(sized(w, h), 0).unwrap();
            gaussian_blur_u8(&src3, &mut cpu, (5, 5), (2.0, 2.0)).unwrap();
            let mut d_dst = Image::<u8, 3>::zeros_cuda(sized(w, h), &stream).unwrap();
            gaussian_blur_u8(&d_src3, &mut d_dst, (5, 5), (2.0, 2.0)).unwrap();
            assert_eq!(
                d_dst.to_host_owned().unwrap().as_slice(),
                cpu.as_slice(),
                "gaussian q8 {w}x{h}"
            );

            // box_blur_u8 (always Q8).
            let mut cpu = Image::<u8, 3>::from_size_val(sized(w, h), 0).unwrap();
            box_blur_u8(&src3, &mut cpu, (3, 5)).unwrap();
            let mut d_dst = Image::<u8, 3>::zeros_cuda(sized(w, h), &stream).unwrap();
            box_blur_u8(&d_src3, &mut d_dst, (3, 5)).unwrap();
            assert_eq!(
                d_dst.to_host_owned().unwrap().as_slice(),
                cpu.as_slice(),
                "box q8 {w}x{h}"
            );
        }
    }
}
