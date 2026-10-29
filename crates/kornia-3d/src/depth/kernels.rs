//! Slice-level CPU kernels for depth unprojection.

use std::mem::MaybeUninit;

use rayon::prelude::*;

use super::{DepthError, DepthSample};
use crate::camera::PinholeCamera;

/// Upper bound on the rayon tasks of one unprojection: the per-task counts
/// and output slices live in stack arrays of this length.
const MAX_TASKS: usize = 64;

/// Unprojection constants of the CPU kernels.
#[derive(Debug, Clone, Copy)]
pub(crate) struct UnprojectParams {
    /// `1 / fx` of the depth camera.
    pub inv_fx: f32,
    /// `1 / fy` of the depth camera.
    pub inv_fy: f32,
    /// Principal point x (pixels).
    pub cx: f32,
    /// Principal point y (pixels).
    pub cy: f32,
    /// Metres per depth unit.
    pub scale: f32,
    /// Depth image width.
    pub width: i32,
    /// Depth image height.
    pub height: i32,
}

// The validity rule is `kornia_imgproc::depth`'s, used as is: a `u16`
// sample is a measurement iff non-zero, an `f32` sample iff finite and
// positive.
//
// The kernels also apply `is_valid_depth_f32` to a value in metres
// (`d · metres_per_unit`): for a finite positive
// scale `s`, the product is valid iff the sample is valid and `d · s`
// neither overflows nor underflows to `0`, and for a `u16` sample (whose
// scale passed [`DepthSample::scale_is_valid`]) iff the sample is non-zero.
pub(crate) use kornia_imgproc::depth::{is_valid_depth_f32, is_valid_depth_u16};

/// Checked `i32` image dimensions.
fn dims_i32(width: usize, height: usize) -> Result<(i32, i32), DepthError> {
    let fits = |v: usize| i32::try_from(v).is_ok();
    if !fits(width) || !fits(height) {
        return Err(DepthError::ImageTooLarge { width, height });
    }
    Ok((width as i32, height as i32))
}

/// Rejects a camera whose `f32` constants would not be finite and non-zero.
fn check_camera(camera: &PinholeCamera) -> Result<(f32, f32, f32, f32), DepthError> {
    let inv_fx = (1.0 / camera.fx) as f32;
    let inv_fy = (1.0 / camera.fy) as f32;
    let (cx, cy) = (camera.cx as f32, camera.cy as f32);
    let finite = [camera.fx as f32, camera.fy as f32, inv_fx, inv_fy, cx, cy]
        .iter()
        .all(|v| v.is_finite());
    if !finite || inv_fx == 0.0 || inv_fy == 0.0 {
        return Err(DepthError::InvalidIntrinsics {
            fx: camera.fx,
            fy: camera.fy,
            cx: camera.cx,
            cy: camera.cy,
        });
    }
    Ok((inv_fx, inv_fy, cx, cy))
}

/// Builds the unprojection constants of a `T` depth image after validating
/// every argument.
pub(crate) fn unproject_params<T: DepthSample>(
    camera: &PinholeCamera,
    metres_per_unit: f32,
    width: usize,
    height: usize,
) -> Result<UnprojectParams, DepthError> {
    if !T::scale_is_valid(metres_per_unit) {
        return Err(DepthError::InvalidDepthScale(metres_per_unit));
    }
    let (inv_fx, inv_fy, cx, cy) = check_camera(camera)?;
    let (w, h) = dims_i32(width, height)?;
    Ok(UnprojectParams {
        inv_fx,
        inv_fy,
        cx,
        cy,
        scale: metres_per_unit,
        width: w,
        height: h,
    })
}

/// Rejects images whose linear pixel index does not fit `u32`.
pub(crate) fn check_pixel_index(width: usize, height: usize) -> Result<(), DepthError> {
    let max_pixels = u64::from(u32::MAX) + 1;
    match width.checked_mul(height) {
        Some(n) if n as u64 <= max_pixels => Ok(()),
        _ => Err(DepthError::ImageTooLarge { width, height }),
    }
}

// ── Unprojection ───────────────────────────────────────────────────────────

/// Number of kept pixels in `depth` (mask non-zero and `valid`).
#[inline(always)]
fn count_kept<T: DepthSample>(
    depth: &[T],
    mask: Option<&[u8]>,
    valid: impl Fn(T) -> bool,
) -> usize {
    match mask {
        // A sum of 0/1 (not `filter().count()`) so the loop vectorises.
        None => depth.iter().map(|&d| usize::from(valid(d))).sum(),
        Some(m) => depth
            .iter()
            .zip(m)
            .map(|(&d, &k)| usize::from((k != 0) & valid(d)))
            .sum(),
    }
}

/// Writes the kept pixels of whole rows `y0..` into `points` (and their
/// linear pixel index into `index`), in row-major order. Returns the number
/// written, or `None` if the slots ran out (never for a matching count).
#[inline(always)]
fn write_rows<T: DepthSample>(
    depth: &[T],
    mask: Option<&[u8]>,
    y0: usize,
    p: &UnprojectParams,
    valid: impl Fn(T) -> bool,
    points: &mut [MaybeUninit<[f64; 3]>],
    mut index: Option<&mut [MaybeUninit<u32>]>,
) -> Option<usize> {
    let w = p.width as usize;
    let mut k = 0usize;
    for (r, row) in depth.chunks_exact(w).enumerate() {
        let y = y0 + r;
        let yn = (y as f32 - p.cy) * p.inv_fy;
        let mrow = mask.map(|m| &m[r * w..(r + 1) * w]);
        for (x, &d) in row.iter().enumerate() {
            if mrow.is_some_and(|m| m[x] == 0) || !valid(d) {
                continue;
            }
            let z = d.metres(p.scale);
            let xn = (x as f32 - p.cx) * p.inv_fx;
            *points.get_mut(k)? =
                MaybeUninit::new([f64::from(xn * z), f64::from(yn * z), f64::from(z)]);
            if let Some(ix) = index.as_deref_mut() {
                *ix.get_mut(k)? = MaybeUninit::new((y * w + x) as u32);
            }
            k += 1;
        }
    }
    Some(k)
}

/// Count and write passes of one band, keeping the samples that
/// [`DepthSample::is_valid_at`] accepts.
#[inline(always)]
fn band_pass<T: DepthSample>(
    depth: &[T],
    mask: Option<&[u8]>,
    y0: usize,
    p: &UnprojectParams,
    out: BandOut<'_>,
) -> Option<usize> {
    let valid = |d: T| d.is_valid_at(p.scale);
    match out {
        BandOut::Count => Some(count_kept(depth, mask, valid)),
        BandOut::Write(pts, idx) => write_rows(depth, mask, y0, p, valid, pts, idx),
    }
}

/// What [`band_pass`] does with a band: count its kept pixels, or write
/// them into the band's output slots.
enum BandOut<'a> {
    Count,
    Write(
        &'a mut [MaybeUninit<[f64; 3]>],
        Option<&'a mut [MaybeUninit<u32>]>,
    ),
}

/// Sequential fallback with the same output as the parallel path. It applies
/// [`is_valid_depth_f32`] to the value in metres for both dtypes, so tests also use it as the oracle for the
/// per-dtype test of the parallel path ([`DepthSample::is_valid_at`]).
pub(crate) fn unproject_sequential<T: DepthSample>(
    depth: &[T],
    mask: Option<&[u8]>,
    p: &UnprojectParams,
    points: &mut Vec<[f64; 3]>,
    mut index: Option<&mut Vec<u32>>,
) {
    points.clear();
    if let Some(ix) = index.as_deref_mut() {
        ix.clear();
    }
    let w = p.width as usize;
    if w == 0 {
        return;
    }
    for (y, row) in depth.chunks_exact(w).enumerate() {
        let yn = (y as f32 - p.cy) * p.inv_fy;
        for (x, &d) in row.iter().enumerate() {
            if mask.is_some_and(|m| m[y * w + x] == 0) {
                continue;
            }
            let z = d.metres(p.scale);
            if !is_valid_depth_f32(z) {
                continue;
            }
            let xn = (x as f32 - p.cx) * p.inv_fx;
            points.push([f64::from(xn * z), f64::from(yn * z), f64::from(z)]);
            if let Some(ix) = index.as_deref_mut() {
                ix.push((y * w + x) as u32);
            }
        }
    }
}

/// Splits the first `total` spare slots of `buf` into `counts.len()`
/// consecutive parts of the given lengths.
fn split_spare<'a, T>(
    buf: &'a mut Vec<T>,
    total: usize,
    counts: &[usize],
    parts: &mut [&'a mut [MaybeUninit<T>]],
) {
    buf.reserve(total);
    let mut rest = &mut buf.spare_capacity_mut()[..total];
    for (part, &n) in parts.iter_mut().zip(counts) {
        let (head, tail) = std::mem::take(&mut rest).split_at_mut(n);
        *part = head;
        rest = tail;
    }
}

/// Unprojects every kept pixel of a row-major `depth` slice into `points`
/// (cleared first), in row-major pixel order, and optionally its linear
/// pixel index into `index`.
///
/// Two parallel passes over at most [`MAX_TASKS`] row bands: count the kept
/// pixels per band, then write each band straight into its own disjoint
/// range of the output's spare capacity, so the output is neither
/// zero-filled nor reallocated when its capacity already suffices.
pub(crate) fn unproject_into<T: DepthSample>(
    depth: &[T],
    mask: Option<&[u8]>,
    p: &UnprojectParams,
    points: &mut Vec<[f64; 3]>,
    mut index: Option<&mut Vec<u32>>,
) {
    points.clear();
    if let Some(ix) = index.as_deref_mut() {
        ix.clear();
    }
    let (w, h) = (p.width as usize, p.height as usize);
    if w == 0 || h == 0 {
        return;
    }
    let rows_per_task = h.div_ceil(MAX_TASKS);
    let band = rows_per_task * w;
    let n_tasks = h.div_ceil(rows_per_task);
    let band_range = |i: usize| (i * band)..((i + 1) * band).min(depth.len());

    let mut counts = [0usize; MAX_TASKS];
    counts[..n_tasks]
        .par_iter_mut()
        .enumerate()
        .for_each(|(i, c)| {
            let r = band_range(i);
            *c = band_pass(&depth[r.clone()], mask.map(|m| &m[r]), 0, p, BandOut::Count)
                .unwrap_or(0);
        });
    let counts = &counts[..n_tasks];
    let total: usize = counts.iter().sum();
    let with_index = index.is_some();

    let all_written = {
        let mut pts_parts: [&mut [MaybeUninit<[f64; 3]>]; MAX_TASKS] =
            std::array::from_fn(|_| Default::default());
        let mut idx_parts: [&mut [MaybeUninit<u32>]; MAX_TASKS] =
            std::array::from_fn(|_| Default::default());
        split_spare(points, total, counts, &mut pts_parts[..n_tasks]);
        if let Some(ix) = index.as_deref_mut() {
            split_spare(ix, total, counts, &mut idx_parts[..n_tasks]);
        }
        pts_parts[..n_tasks]
            .par_iter_mut()
            .zip(idx_parts[..n_tasks].par_iter_mut())
            .enumerate()
            .all(|(i, (pts, idx))| {
                let r = band_range(i);
                let want = pts.len();
                let idx = if with_index { Some(&mut **idx) } else { None };
                band_pass(
                    &depth[r.clone()],
                    mask.map(|m| &m[r]),
                    i * rows_per_task,
                    p,
                    BandOut::Write(pts, idx),
                ) == Some(want)
            })
    };

    if !all_written {
        // Unreachable for an immutable input (both passes apply the same
        // predicate to the same samples); kept so `set_len` below is only
        // ever reached with every slot written.
        unproject_sequential(depth, mask, p, points, index);
        return;
    }
    // SAFETY: the parts tile `spare_capacity_mut()[..total]` exactly (their
    // lengths are `counts`, which sum to `total`), and every band wrote all
    // of its slots (`write_rows` returned the part length for each one), so
    // the first `total` elements are initialized. `reserve(total)` above
    // guarantees the capacity.
    unsafe { points.set_len(total) };
    if let Some(ix) = index {
        // SAFETY: same tiling and full-write argument as for `points`; the
        // index parts were split with the same `counts` and each band wrote
        // an index for every point it wrote.
        unsafe { ix.set_len(total) };
    }
}
