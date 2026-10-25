//! Slice-level CPU kernels for depth unprojection and registration.
//!
//! Every float expression here has a textual twin in `cuda.rs`'s `KERNELS`,
//! which nvrtc compiles with `fmad=false` (no contraction, IEEE division,
//! no flush-to-zero), so the host and the device round every step the same
//! way and produce identical points and depth values.

use std::mem::MaybeUninit;
use std::sync::atomic::{AtomicU16, AtomicU32, Ordering};

use kornia_image::ImageSize;
use rayon::prelude::*;

use super::{DepthError, DepthSample};
use crate::camera::PinholeCamera;
use crate::pose::Pose3d;

/// `DepthSample::KERNEL_INDEX` of `u16`.
pub(crate) const KIND_U16: usize = 0;
/// `DepthSample::KERNEL_INDEX` of `f32`.
pub(crate) const KIND_F32: usize = 1;

/// Largest accepted splat radius. A wider footprint is a hole-filling
/// filter rather than a splat, and the bound keeps every pixel coordinate
/// plus radius inside `i32` on both host and device.
pub const MAX_SPLAT_RADIUS: usize = 32;

/// Upper bound on the rayon tasks of one unprojection: the per-task counts
/// and output slices live in stack arrays of this length.
const MAX_TASKS: usize = 64;

/// Source rows per rayon task in the registration scatter.
const SCATTER_ROWS_PER_TASK: usize = 16;

/// Elements per rayon task in the z-buffer fill and finalize passes.
const FILL_CHUNK: usize = 1 << 16;

/// Largest representable `u16` output; `u16::MAX` is the inverted-key
/// sentinel and is never written.
const U16_MAX_OUTPUT: f32 = 65534.0;

/// Unprojection constants, shared verbatim by the CPU kernels and the CUDA
/// kernel ABI (`struct Unp` in `cuda.rs`).
#[repr(C)]
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

/// Registration constants, shared verbatim by the CPU kernels and the CUDA
/// kernel ABI (`struct Reg` in `cuda.rs`).
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub(crate) struct RegisterParams {
    /// Source camera and image.
    pub src: UnprojectParams,
    /// Output depth units per metre (`1 / scale`).
    pub inv_scale: f32,
    /// `dst_from_src` rotation, row-major.
    pub r: [f32; 9],
    /// `dst_from_src` translation, metres.
    pub t: [f32; 3],
    /// Destination focal length x.
    pub fx: f32,
    /// Destination focal length y.
    pub fy: f32,
    /// Destination principal point x.
    pub cx: f32,
    /// Destination principal point y.
    pub cy: f32,
    /// Destination width.
    pub width: i32,
    /// Destination height.
    pub height: i32,
    /// Splat radius in destination pixels.
    pub radius: i32,
}

/// A depth sample converted to metres is a measurement iff it is positive
/// and finite: `0` and `NaN` (and negative or infinite values) are missing.
#[inline(always)]
pub(crate) fn is_valid_metres(z: f32) -> bool {
    z > 0.0 && z < f32::INFINITY
}

/// Checked `i32` image dimensions with room for a splat radius on top.
fn dims_i32(width: usize, height: usize) -> Result<(i32, i32), DepthError> {
    let fits = |v: usize| {
        v.checked_add(MAX_SPLAT_RADIUS)
            .is_some_and(|s| i32::try_from(s).is_ok())
    };
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

/// Builds the unprojection constants after validating every argument.
pub(crate) fn unproject_params(
    camera: &PinholeCamera,
    metres_per_unit: f32,
    width: usize,
    height: usize,
) -> Result<UnprojectParams, DepthError> {
    if !(metres_per_unit.is_finite() && metres_per_unit > 0.0) {
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

/// Builds the registration constants after validating every argument.
pub(crate) fn register_params(
    src_camera: &PinholeCamera,
    metres_per_unit: f32,
    dst_from_src: &Pose3d,
    dst_camera: &PinholeCamera,
    splat_radius: usize,
    src_size: ImageSize,
    dst_size: ImageSize,
) -> Result<RegisterParams, DepthError> {
    let src = unproject_params(src_camera, metres_per_unit, src_size.width, src_size.height)?;
    let inv_scale = 1.0f32 / metres_per_unit;
    if !inv_scale.is_finite() {
        return Err(DepthError::InvalidDepthScale(metres_per_unit));
    }
    if splat_radius > MAX_SPLAT_RADIUS {
        return Err(DepthError::SplatRadiusTooLarge {
            radius: splat_radius,
            max: MAX_SPLAT_RADIUS,
        });
    }
    // Validates the destination focal lengths; the reciprocals are unused.
    check_camera(dst_camera)?;
    let (dst_w, dst_h) = dims_i32(dst_size.width, dst_size.height)?;
    // Column-major [c0.x, c0.y, c0.z, c1.x, ...]; r[3 * i + j] = R[i][j].
    let m = dst_from_src.rotation.to_cols_array();
    let mut r = [0.0f32; 9];
    for i in 0..3 {
        for j in 0..3 {
            r[3 * i + j] = m[3 * j + i] as f32;
        }
    }
    let tr = dst_from_src.translation;
    let t = [tr.x as f32, tr.y as f32, tr.z as f32];
    if !r.iter().chain(t.iter()).all(|v| v.is_finite()) {
        return Err(DepthError::InvalidTransform);
    }
    Ok(RegisterParams {
        src,
        inv_scale,
        r,
        t,
        fx: dst_camera.fx as f32,
        fy: dst_camera.fy as f32,
        cx: dst_camera.cx as f32,
        cy: dst_camera.cy as f32,
        width: dst_w,
        height: dst_h,
        radius: splat_radius as i32,
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

/// Number of kept pixels in `depth` (mask non-zero and valid metres).
fn count_kept<T: DepthSample>(depth: &[T], mask: Option<&[u8]>, scale: f32) -> usize {
    match mask {
        None => depth
            .iter()
            .filter(|&&d| is_valid_metres(d.metres(scale)))
            .count(),
        Some(m) => depth
            .iter()
            .zip(m)
            .filter(|&(&d, &k)| k != 0 && is_valid_metres(d.metres(scale)))
            .count(),
    }
}

/// Writes the kept pixels of whole rows `y0..` into `points` (and their
/// linear pixel index into `index`), in row-major order. Returns the number
/// written, or `None` if the slots ran out (never for a matching count).
fn write_rows<T: DepthSample>(
    depth: &[T],
    mask: Option<&[u8]>,
    y0: usize,
    p: &UnprojectParams,
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
            if mrow.is_some_and(|m| m[x] == 0) {
                continue;
            }
            let z = d.metres(p.scale);
            if !is_valid_metres(z) {
                continue;
            }
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

/// Sequential fallback with the same output as the parallel path.
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
            if !is_valid_metres(z) {
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
            *c = count_kept(&depth[r.clone()], mask.map(|m| &m[r]), p.scale);
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
                write_rows(
                    &depth[r.clone()],
                    mask.map(|m| &m[r]),
                    i * rows_per_task,
                    p,
                    pts,
                    idx,
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

// ── Registration ───────────────────────────────────────────────────────────

/// Projects source pixel `(x, y)` at depth `z` metres into the destination
/// camera. Returns the destination depth `qz` (metres) and the inclusive,
/// image-clipped splat rectangle `[x0, x1, y0, y1]`, or `None` when the
/// point is behind the destination camera or its splat misses the image.
#[inline(always)]
fn splat_target(p: &RegisterParams, x: usize, y: usize, z: f32) -> Option<(f32, [usize; 4])> {
    let s = &p.src;
    let px = ((x as f32 - s.cx) * s.inv_fx) * z;
    let py = ((y as f32 - s.cy) * s.inv_fy) * z;
    let (r, t) = (&p.r, &p.t);
    let qx = r[0] * px + r[1] * py + r[2] * z + t[0];
    let qy = r[3] * px + r[4] * py + r[5] * z + t[1];
    let qz = r[6] * px + r[7] * py + r[8] * z + t[2];
    if qz.is_nan() || qz <= 0.0 {
        return None;
    }
    let iz = 1.0 / qz;
    let u = p.fx * qx * iz + p.cx;
    let v = p.fy * qy * iz + p.cy;
    let uf = (u + 0.5).floor();
    let vf = (v + 0.5).floor();
    let rr = p.radius as f32;
    if !((-rr..(p.width + p.radius) as f32).contains(&uf)
        && (-rr..(p.height + p.radius) as f32).contains(&vf))
    {
        return None;
    }
    let (ui, vi) = (uf as i32, vf as i32);
    let x0 = (ui - p.radius).max(0) as usize;
    let x1 = (ui + p.radius).min(p.width - 1) as usize;
    let y0 = (vi - p.radius).max(0) as usize;
    let y1 = (vi + p.radius).min(p.height - 1) as usize;
    Some((qz, [x0, x1, y0, y1]))
}

/// Destination `u16` value of a point at `qz` metres, or `None` when it
/// rounds outside `1..=65534` (`0` is "empty", `65535` the key sentinel).
#[inline(always)]
fn quantise_u16(qz: f32, inv_scale: f32) -> Option<u16> {
    let q = (qz * inv_scale).round();
    if (1.0..=U16_MAX_OUTPUT).contains(&q) {
        Some(q as u16)
    } else {
        None
    }
}

/// Destination `f32` value of a point at `qz` metres, or `None` when it is
/// not a valid measurement in the output units.
#[inline(always)]
fn scale_f32(qz: f32, inv_scale: f32) -> Option<f32> {
    let o = qz * inv_scale;
    is_valid_metres(o).then_some(o)
}

/// Calls `visit(qz, rect)` for every valid source pixel that lands in the
/// destination, in parallel over bands of source rows.
fn scatter_rows<T: DepthSample>(
    src: &[T],
    p: &RegisterParams,
    visit: impl Fn(f32, [usize; 4]) + Sync,
) {
    let w = p.src.width as usize;
    if w == 0 || p.src.height == 0 || p.width == 0 || p.height == 0 {
        return;
    }
    src.par_chunks(w * SCATTER_ROWS_PER_TASK)
        .enumerate()
        .for_each(|(i, band)| {
            for (r, row) in band.chunks_exact(w).enumerate() {
                let y = i * SCATTER_ROWS_PER_TASK + r;
                for (x, &d) in row.iter().enumerate() {
                    let z = d.metres(p.src.scale);
                    if !is_valid_metres(z) {
                        continue;
                    }
                    if let Some((qz, rect)) = splat_target(p, x, y, z) {
                        visit(qz, rect);
                    }
                }
            }
        });
}

/// Applies `key` to every cell of the splat rectangle: keeps the largest
/// key, i.e. the nearest point (keys are inverted depths).
#[inline(always)]
fn splat_max<A, K: Copy + PartialOrd>(
    zbuf: &[A],
    width: usize,
    rect: [usize; 4],
    key: K,
    load: impl Fn(&A) -> K,
    fetch_max: impl Fn(&A, K),
) {
    let [x0, x1, y0, y1] = rect;
    for yy in y0..=y1 {
        for cell in &zbuf[yy * width + x0..=yy * width + x1] {
            // Keys only grow, so a stale load can only under-estimate the
            // cell and never skips a needed update.
            if key > load(cell) {
                fetch_max(cell, key);
            }
        }
    }
}

const _: () = assert!(
    std::mem::size_of::<AtomicU16>() == std::mem::size_of::<u16>()
        && std::mem::align_of::<AtomicU16>() == std::mem::align_of::<u16>()
);
const _: () = assert!(
    std::mem::size_of::<AtomicU32>() == std::mem::size_of::<f32>()
        && std::mem::align_of::<AtomicU32>() == std::mem::align_of::<f32>()
);

/// Views a `u16` buffer as atomics for the parallel z-buffer.
fn as_atomic_u16(buf: &mut [u16]) -> &[AtomicU16] {
    // SAFETY: `AtomicU16` has the same size and bit validity as `u16`, and
    // the const assertion above pins its alignment to `u16`'s. The exclusive
    // borrow of `buf` lasts as long as the returned slice, so no non-atomic
    // access aliases it meanwhile.
    unsafe { std::slice::from_raw_parts(buf.as_mut_ptr().cast::<AtomicU16>(), buf.len()) }
}

/// Views an `f32` buffer as `u32` atomics over its bit patterns.
fn as_atomic_bits(buf: &mut [f32]) -> &[AtomicU32] {
    // SAFETY: `AtomicU32` has the size and alignment of `f32` (const
    // assertion above) and every bit pattern is a valid `u32`; the exclusive
    // borrow of `buf` lasts as long as the returned slice.
    unsafe { std::slice::from_raw_parts(buf.as_mut_ptr().cast::<AtomicU32>(), buf.len()) }
}

/// Registers a `u16` depth image. The z-buffer is `dst` itself, holding
/// `!value` (so nearer = larger key, empty = `0` = the initial fill) and
/// updated with `fetch_max`; the finalize pass inverts the keys back.
pub(crate) fn register_u16(src: &[u16], dst: &mut [u16], p: &RegisterParams) {
    dst.par_chunks_mut(FILL_CHUNK).for_each(|c| c.fill(0));
    let width = p.width as usize;
    {
        let zbuf = as_atomic_u16(dst);
        scatter_rows(src, p, |qz, rect| {
            let Some(v) = quantise_u16(qz, p.inv_scale) else {
                return;
            };
            splat_max(
                zbuf,
                width,
                rect,
                !v,
                |c| c.load(Ordering::Relaxed),
                |c, k| {
                    c.fetch_max(k, Ordering::Relaxed);
                },
            );
        });
    }
    dst.par_chunks_mut(FILL_CHUNK).for_each(|c| {
        for d in c {
            if *d != 0 {
                *d = !*d;
            }
        }
    });
}

/// Registers an `f32` depth image. The z-buffer is `dst`'s bit patterns,
/// holding `!bits(value)`: positive finite floats order like their bits,
/// so nearer = larger key and empty = `0` = the bits of the `0.0` fill.
pub(crate) fn register_f32(src: &[f32], dst: &mut [f32], p: &RegisterParams) {
    dst.par_chunks_mut(FILL_CHUNK).for_each(|c| c.fill(0.0));
    let width = p.width as usize;
    {
        let zbuf = as_atomic_bits(dst);
        scatter_rows(src, p, |qz, rect| {
            let Some(o) = scale_f32(qz, p.inv_scale) else {
                return;
            };
            splat_max(
                zbuf,
                width,
                rect,
                !o.to_bits(),
                |c| c.load(Ordering::Relaxed),
                |c, k| {
                    c.fetch_max(k, Ordering::Relaxed);
                },
            );
        });
    }
    dst.par_chunks_mut(FILL_CHUNK).for_each(|c| {
        for d in c {
            let k = d.to_bits();
            *d = f32::from_bits(if k != 0 { !k } else { 0 });
        }
    });
}
