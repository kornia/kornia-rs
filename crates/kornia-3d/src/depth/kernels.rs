//! Slice-level CPU kernels for depth unprojection and registration.

use std::mem::MaybeUninit;
use std::sync::atomic::{AtomicU16, AtomicU32, Ordering};

use kornia_imgproc::simd::cpu_features;
use rayon::prelude::*;

use super::{DepthError, DepthImage, DepthSample};
use crate::camera::PinholeCamera;
use crate::pose::Pose3d;

/// Largest accepted splat radius. A wider footprint is a hole-filling
/// filter rather than a splat, and the bound keeps every pixel coordinate
/// plus radius inside `i32`.
pub const MAX_SPLAT_RADIUS: usize = 32;

/// Upper bound on the rayon tasks of one unprojection: the per-task counts
/// and output slices live in stack arrays of this length.
const MAX_TASKS: usize = 64;

/// Fewest source rows per band of the registration scatter (see
/// [`scatter_rows`]).
const MIN_BAND_ROWS: usize = 16;

/// Fewest bands a slant-sized band layout may have (see [`band_layout`]).
const MIN_BANDS: usize = 4;

/// Destination rows a splat may stray beyond the slant of its band and
/// still stay in the band's window (see [`band_layout`]).
const BAND_MARGIN_ROWS: f64 = 8.0;

/// Deferred splats per rayon task in the registration scatter.
const DEFERRED_CHUNK: usize = 4096;

/// Elements per rayon task in the z-buffer fill and finalize passes.
const FILL_CHUNK: usize = 1 << 16;

/// Largest representable `u16` output; `u16::MAX` is the inverted-key
/// sentinel and is never written.
const U16_MAX_OUTPUT: f32 = 65534.0;

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
    /// Metres per depth unit, `1 / meter` in `f32`.
    pub scale: f32,
    /// Depth image width.
    pub width: i32,
    /// Depth image height.
    pub height: i32,
}

/// Registration constants of the CPU kernels.
#[derive(Debug, Clone, Copy)]
pub(crate) struct RegisterParams {
    /// Source camera and image; `src.scale` is `1 / meter` of the source.
    pub src: UnprojectParams,
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
    /// `1 / meter` of the destination in `f32`: the output value is
    /// `z_dst / dst_scale`.
    pub dst_scale: f32,
}

// The validity rule is `kornia_imgproc::depth`'s, used as is: a `u16`
// sample is a measurement iff non-zero, an `f32` sample iff finite and
// positive.
//
// The kernels also apply `is_valid_depth_f32` to a value in metres
// (`d · s`, `s = 1 / meter`) and to an output value: for a finite
// positive `s`, the product is valid iff the sample is valid and `d · s` neither overflows nor
// underflows to `0`, and for a `u16` sample (whose `meter` passed
// [`DepthSample::meter_is_valid`]) iff the sample is non-zero.
pub(crate) use kornia_imgproc::depth::{is_valid_depth_f32, is_valid_depth_u16};

/// Image dimensions checked to fit `i32` with room for a splat radius on
/// top.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct Dims {
    /// Width in pixels.
    pub width: i32,
    /// Height in pixels.
    pub height: i32,
}

impl Dims {
    /// Checks that both sides plus [`MAX_SPLAT_RADIUS`] fit `i32`.
    fn new(width: usize, height: usize) -> Result<Self, DepthError> {
        let fits = |v: usize| {
            v.checked_add(MAX_SPLAT_RADIUS)
                .is_some_and(|s| i32::try_from(s).is_ok())
        };
        if !fits(width) || !fits(height) {
            return Err(DepthError::ImageTooLarge { width, height });
        }
        Ok(Self {
            width: width as i32,
            height: height as i32,
        })
    }
}

/// The `f32` pinhole constants of the CPU kernels: reciprocal focal lengths
/// and principal point, all finite and the reciprocals non-zero.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct Intrinsics32 {
    /// `1 / fx`.
    pub inv_fx: f32,
    /// `1 / fy`.
    pub inv_fy: f32,
    /// Principal point x (pixels).
    pub cx: f32,
    /// Principal point y (pixels).
    pub cy: f32,
}

impl Intrinsics32 {
    /// Rejects a camera whose `f32` constants would not be finite and
    /// non-zero.
    fn new(camera: &PinholeCamera) -> Result<Self, DepthError> {
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
        Ok(Self {
            inv_fx,
            inv_fy,
            cx,
            cy,
        })
    }
}

/// Builds the unprojection constants of `depth` after validating the
/// camera and the image size; `depth`'s `meter` was validated by its
/// constructor.
pub(crate) fn unproject_params<T: DepthSample>(
    camera: &PinholeCamera,
    depth: &DepthImage<T>,
) -> Result<UnprojectParams, DepthError> {
    let k = Intrinsics32::new(camera)?;
    let dims = Dims::new(depth.width(), depth.height())?;
    Ok(UnprojectParams {
        inv_fx: k.inv_fx,
        inv_fy: k.inv_fy,
        cx: k.cx,
        cy: k.cy,
        scale: depth.metres_per_unit(),
        width: dims.width,
        height: dims.height,
    })
}

/// Builds the registration constants of `src` into `dst` after validating
/// the cameras, the transform, the splat radius and both image sizes; both
/// `meter`s were validated by their constructors.
pub(crate) fn register_params<T: DepthSample>(
    src: &DepthImage<T>,
    src_camera: &PinholeCamera,
    dst_from_src: &Pose3d,
    dst_camera: &PinholeCamera,
    splat_radius: usize,
    dst: &DepthImage<T>,
) -> Result<RegisterParams, DepthError> {
    let src_params = unproject_params(src_camera, src)?;
    if splat_radius > MAX_SPLAT_RADIUS {
        return Err(DepthError::SplatRadiusTooLarge {
            radius: splat_radius,
            max: MAX_SPLAT_RADIUS,
        });
    }
    // Validates the destination focal lengths; the reciprocals are unused.
    Intrinsics32::new(dst_camera)?;
    let dst_dims = Dims::new(dst.width(), dst.height())?;
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
        src: src_params,
        r,
        t,
        fx: dst_camera.fx as f32,
        fy: dst_camera.fy as f32,
        cx: dst_camera.cx as f32,
        cy: dst_camera.cy as f32,
        width: dst_dims.width,
        height: dst_dims.height,
        radius: splat_radius as i32,
        dst_scale: dst.metres_per_unit(),
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

// ── Registration ───────────────────────────────────────────────────────────

/// Source pixels projected per batch of the registration scatter: a batch
/// is projected branch-free (NEON on aarch64), then its splats are applied.
pub(crate) const PROJECT_BATCH: usize = 64;

/// Destination cells of one batch: z-buffer key (`0` = no splat) and the
/// nearest destination pixel of each source pixel.
pub(crate) struct Batch {
    pub key: [u32; PROJECT_BATCH],
    pub u: [i32; PROJECT_BATCH],
    pub v: [i32; PROJECT_BATCH],
}

impl Batch {
    pub(crate) fn new() -> Self {
        Self {
            key: [0; PROJECT_BATCH],
            u: [0; PROJECT_BATCH],
            v: [0; PROJECT_BATCH],
        }
    }
}

/// Destination key and nearest destination pixel of source pixel `x` of a
/// row whose `(y - cy) / fy` is `yn`, at depth `z` metres. The key is `0`
/// unless `z` is valid, the point is in front of the destination camera,
/// its splat touches the image and its value is representable (`key_u16` /
/// `key_f32`). [`project_neon`] is its 4-lane twin.
///
/// This restates the pinhole model of `camera::pinhole::project_point`
/// instead of calling it: that function works in `f64` through `Pose3d` and
/// `Vec3F64`, with its own `z <= 1e-8` cut, whereas the scalar and NEON
/// paths here must evaluate the same `f32` operations in the same order to
/// stay bit-identical.
#[inline(always)]
fn project_one(p: &RegisterParams, x: u32, yn: f32, z: f32, u16_keys: bool) -> Splat {
    let s = &p.src;
    let (r, t) = (&p.r, &p.t);
    let px = ((x as f32 - s.cx) * s.inv_fx) * z;
    let py = yn * z;
    let qx = r[0] * px + r[1] * py + r[2] * z + t[0];
    let qy = r[3] * px + r[4] * py + r[5] * z + t[1];
    let qz = r[6] * px + r[7] * py + r[8] * z + t[2];
    let iz = 1.0 / qz;
    let uf = (p.fx * qx * iz + p.cx + 0.5).floor();
    let vf = (p.fy * qy * iz + p.cy + 0.5).floor();
    let rr = p.radius as f32;
    let inside = (uf >= -rr)
        & (uf < (p.width + p.radius) as f32)
        & (vf >= -rr)
        & (vf < (p.height + p.radius) as f32);
    // `qz > 0` is false for NaN.
    let ok = is_valid_depth_f32(z) & (qz > 0.0) & inside;
    let k = if u16_keys {
        key_u16(qz, p.dst_scale)
    } else {
        key_f32(qz, p.dst_scale)
    };
    Splat {
        u: uf as i32,
        v: vf as i32,
        key: if ok { k } else { 0 },
    }
}

/// Projects `row`, source pixels `x0..x0 + row.len()` of row `y` (at most
/// [`PROJECT_BATCH`]), into `batch`: `key[j]`, `u[j]`, `v[j]` as
/// [`project_one`] (`u`, `v` are meaningful only for a non-zero key).
#[inline(always)]
pub(crate) fn project_batch<T: DepthSample>(
    row: &[T],
    x0: usize,
    y: usize,
    p: &RegisterParams,
    u16_keys: bool,
    neon: bool,
    batch: &mut Batch,
) {
    let row = &row[..row.len().min(PROJECT_BATCH)];
    let mut zs = [0.0f32; PROJECT_BATCH];
    for (z, &d) in zs.iter_mut().zip(row) {
        *z = d.metres(p.src.scale);
    }
    let zs = &zs[..row.len()];
    let yn = (y as f32 - p.src.cy) * p.src.inv_fy;
    #[cfg(target_arch = "aarch64")]
    let done = if neon {
        // SAFETY: `neon` is `cpu_features().has_neon`; `zs.len()` is at most
        // `PROJECT_BATCH`, the length of every `batch` array.
        unsafe { project_neon(zs, x0, yn, p, u16_keys, batch) }
    } else {
        0
    };
    #[cfg(not(target_arch = "aarch64"))]
    let done = {
        let _ = neon;
        0
    };
    for (j, &z) in zs.iter().enumerate().skip(done) {
        let sp = project_one(p, (x0 + j) as u32, yn, z, u16_keys);
        batch.key[j] = sp.key;
        batch.u[j] = sp.u;
        batch.v[j] = sp.v;
    }
}

/// NEON twin of [`project_one`] over 4 lanes: the same IEEE operations in
/// the same order (separate multiply and add, no fusion), so every lane
/// equals the scalar result bit for bit. Projects the first multiple of 4
/// of `zs` and returns how many it did.
///
/// # Safety
///
/// NEON must be available and `zs.len() <= PROJECT_BATCH`.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn project_neon(
    zs: &[f32],
    x0: usize,
    yn: f32,
    p: &RegisterParams,
    u16_keys: bool,
    batch: &mut Batch,
) -> usize {
    use std::arch::aarch64::*;
    let s = &p.src;
    let (scx, sifx, vyn) = (vdupq_n_f32(s.cx), vdupq_n_f32(s.inv_fx), vdupq_n_f32(yn));
    let r = p.r.map(|v| vdupq_n_f32(v));
    let t = p.t.map(|v| vdupq_n_f32(v));
    let (fx, fy) = (vdupq_n_f32(p.fx), vdupq_n_f32(p.fy));
    let (cx, cy) = (vdupq_n_f32(p.cx), vdupq_n_f32(p.cy));
    let (zero, half, one) = (vdupq_n_f32(0.0), vdupq_n_f32(0.5), vdupq_n_f32(1.0));
    let inf = vdupq_n_f32(f32::INFINITY);
    let neg_rr = vdupq_n_f32(-(p.radius as f32));
    let u_end = vdupq_n_f32((p.width + p.radius) as f32);
    let v_end = vdupq_n_f32((p.height + p.radius) as f32);
    let dst_scale = vdupq_n_f32(p.dst_scale);
    let u16_max = vdupq_n_f32(U16_MAX_OUTPUT);
    let lanes = vld1q_u32([0u32, 1, 2, 3].as_ptr());
    let n = zs.len() & !3;
    let mut j = 0;
    while j < n {
        let z = vld1q_f32(zs.as_ptr().add(j));
        let xf = vcvtq_f32_u32(vaddq_u32(vdupq_n_u32((x0 + j) as u32), lanes));
        let px = vmulq_f32(vmulq_f32(vsubq_f32(xf, scx), sifx), z);
        let py = vmulq_f32(vyn, z);
        // ((r0 · px + r1 · py) + r2 · z) + t, as the scalar expression.
        let qx = vaddq_f32(
            vaddq_f32(
                vaddq_f32(vmulq_f32(r[0], px), vmulq_f32(r[1], py)),
                vmulq_f32(r[2], z),
            ),
            t[0],
        );
        let qy = vaddq_f32(
            vaddq_f32(
                vaddq_f32(vmulq_f32(r[3], px), vmulq_f32(r[4], py)),
                vmulq_f32(r[5], z),
            ),
            t[1],
        );
        let qz = vaddq_f32(
            vaddq_f32(
                vaddq_f32(vmulq_f32(r[6], px), vmulq_f32(r[7], py)),
                vmulq_f32(r[8], z),
            ),
            t[2],
        );
        let iz = vdivq_f32(one, qz);
        let uf = vrndmq_f32(vaddq_f32(
            vaddq_f32(vmulq_f32(vmulq_f32(fx, qx), iz), cx),
            half,
        ));
        let vf = vrndmq_f32(vaddq_f32(
            vaddq_f32(vmulq_f32(vmulq_f32(fy, qy), iz), cy),
            half,
        ));
        let inside = vandq_u32(
            vandq_u32(vcgeq_f32(uf, neg_rr), vcltq_f32(uf, u_end)),
            vandq_u32(vcgeq_f32(vf, neg_rr), vcltq_f32(vf, v_end)),
        );
        // is_valid_depth_f32(z): for z > 0, finite is z < inf.
        let ok = vandq_u32(
            vandq_u32(vcgtq_f32(z, zero), vcltq_f32(z, inf)),
            vandq_u32(vcgtq_f32(qz, zero), inside),
        );
        let key = if u16_keys {
            // key_u16: round half away from zero, keep 1..=65534, !value.
            let q = vrndaq_f32(vdivq_f32(qz, dst_scale));
            let in_range = vandq_u32(vcgeq_f32(q, one), vcleq_f32(q, u16_max));
            let k = veorq_u32(vcvtq_u32_f32(q), vdupq_n_u32(0xffff));
            vandq_u32(k, vandq_u32(ok, in_range))
        } else {
            // key_f32: valid output, !bits(value).
            let o = vdivq_f32(qz, dst_scale);
            let valid = vandq_u32(vcgtq_f32(o, zero), vcltq_f32(o, inf));
            let k = vmvnq_u32(vreinterpretq_u32_f32(o));
            vandq_u32(k, vandq_u32(ok, valid))
        };
        vst1q_u32(batch.key.as_mut_ptr().add(j), key);
        vst1q_s32(batch.u.as_mut_ptr().add(j), vcvtq_s32_f32(uf));
        vst1q_s32(batch.v.as_mut_ptr().add(j), vcvtq_s32_f32(vf));
        j += 4;
    }
    n
}

/// Destination `u16` z-buffer key of a point at `qz` metres, `scale` metres
/// per destination unit (`1 / meter` of `dst`): `!value` with
/// `value = round(qz / scale)`, or `0` when the
/// value is outside `1..=65534` (`0` is "empty", `65535` would collide with
/// the empty key).
#[inline(always)]
fn key_u16(qz: f32, scale: f32) -> u32 {
    let q = (qz / scale).round();
    if (1.0..=U16_MAX_OUTPUT).contains(&q) {
        u32::from(!(q as u16))
    } else {
        0
    }
}

/// Destination `f32` z-buffer key of a point at `qz` metres, `scale` metres
/// per destination unit (`1 / meter` of `dst`, rounded to `f32`):
/// `!bits(value)` with `value = qz / scale`, or `0` when the value is not a
/// valid measurement. A division by the same `scale` the unprojection
/// multiplied by, rather than a multiplication by `meter`: it is correctly
/// rounded, so an identity registration with equal `meter`s returns
/// `(d · scale) / scale`, which is `d` for far more samples than
/// `(d · scale) · meter` because `scale` is itself rounded (`1 / 1000` is
/// not exact): for millimetres, every `f32` in `[300, 12000)` round-trips
/// 97.8% vs 41.1% of the time, and of the integers `1..=65535` 1510 vs
/// 38588 are 1 ulp off (pinned by test); never fewer at the other `meter`s
/// tried. Exactly `d` is not achievable in general,
/// as `d · scale` is itself rounded. Positive finite floats order like
/// their bits, so nearer = larger key.
#[inline(always)]
fn key_f32(qz: f32, scale: f32) -> u32 {
    let o = qz / scale;
    if is_valid_depth_f32(o) {
        !o.to_bits()
    } else {
        0
    }
}

/// Splat of one projected source pixel: nearest destination pixel `(u, v)`
/// and z-buffer key (`0` = nothing to splat).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct Splat {
    /// Destination column.
    u: i32,
    /// Destination row.
    v: i32,
    /// Inverted depth key; larger = nearer.
    key: u32,
}

/// Splats `key` around `(u, v)` (the clipped `(2r + 1)²` square): every
/// cell whose key is smaller gets `update(cell, key)`, so the largest key,
/// i.e. the nearest point, wins. `update` is a plain store when no other
/// thread touches these cells meanwhile, an atomic `fetch_max` otherwise
/// (keys only grow, so a stale `load` never skips a needed update).
#[inline(always)]
fn splat<A>(
    zbuf: &[A],
    p: &RegisterParams,
    Splat { u, v, key }: Splat,
    load: &impl Fn(&A) -> u32,
    update: &impl Fn(&A, u32),
) {
    let (dw, radius) = (p.width as usize, p.radius);
    let put = |cell: &A| {
        if key > load(cell) {
            update(cell, key);
        }
    };
    if radius == 0 {
        // `inside` with a zero radius bounds (u, v) to the image.
        put(&zbuf[v as usize * dw + u as usize]);
        return;
    }
    let x0 = (u - radius).max(0) as usize;
    let x1 = (u + radius).min(p.width - 1) as usize;
    let y0 = (v - radius).max(0) as usize;
    let y1 = (v + radius).min(p.height - 1) as usize;
    for yy in y0..=y1 {
        zbuf[yy * dw + x0..=yy * dw + x1].iter().for_each(put);
    }
}

/// Band layout of [`scatter_rows`] (see [`band_layout`]).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct BandLayout {
    /// Destination-row shift of the band windows.
    offset: i64,
    /// Source rows per band.
    band_rows: usize,
}

/// Band layout of [`scatter_rows`] from the rotation of `p`: the
/// destination-row shift of the source principal ray (relative to plain
/// row rescaling), which centres the windows on where the bands land, and
/// the source rows per band, tall enough that a window covers the slant of
/// a rotated (rolled) row plus the splat radius. Only a scheduling choice:
/// splats that still leave their window are deferred, so any layout gives
/// the same result.
fn band_layout(p: &RegisterParams) -> BandLayout {
    let (sh, dh) = (f64::from(p.src.height), f64::from(p.height));
    let r = p.r.map(f64::from);
    let (fy, cy) = (f64::from(p.fy), f64::from(p.cy));
    // Destination row of the direction through source pixel (x, cy).
    let v_at = |x: f64| {
        let xn = (x - f64::from(p.src.cx)) * f64::from(p.src.inv_fx);
        fy * (r[3] * xn + r[5]) / (r[6] * xn + r[8]) + cy
    };
    let shift = v_at(f64::from(p.src.cx)) - f64::from(p.src.cy) * dh / sh;
    let offset = if r[8] > 0.0 && shift.abs() < dh {
        shift.round() as i64
    } else {
        0
    };
    let slant = (v_at(0.0) - v_at(f64::from(p.src.width - 1))).abs();
    // Window half-height (half a destination slice) >= slant / 2 + radius
    // + margin.
    let slice = slant + f64::from(2 * p.radius) + 2.0 * BAND_MARGIN_ROWS;
    let rows = (slice * sh / dh).ceil();
    // Fewer than MIN_BANDS bands would serialise the scatter: keep thin
    // bands then and let the deferred (atomic, parallel) path take the
    // splats that leave their windows.
    let rows = if rows.is_finite() && rows * MIN_BANDS as f64 <= sh {
        (rows as usize).max(MIN_BAND_ROWS)
    } else {
        MIN_BAND_ROWS
    };
    BandLayout {
        offset,
        band_rows: rows,
    }
}

/// Splats every valid source pixel into the z-buffer `zbuf`, whose keys
/// only grow (larger key = nearer point).
///
/// The source rows are cut into `nb` bands (height from [`band_layout`])
/// and the destination rows into `nb` slices; band `i` owns the window of
/// slice `i` widened by half a slice on each side (and shifted by the
/// [`band_layout`] offset), so the windows of bands `i` and `i + 2` never
/// overlap.
/// The even bands run in parallel, then the odd ones, each writing its own
/// window with plain load/store (no atomic read-modify-write). Splats that
/// leave their band's window (strong rotations, near-field parallax) are
/// deferred and applied at the end, in parallel with atomic `fetch_max`.
/// The result is the per-cell maximum key, independent of scheduling.
///
/// Deferred splats cannot be applied while a phase runs (a concurrent plain
/// store of the owning band could overwrite them), so each band keeps them
/// in its own buffer until both phases end. A band that defers nothing
/// allocates nothing, which is the common case of a small baseline; at
/// worst the buffers hold one splat per valid source pixel, and they are
/// freed when the call returns (the free-function API has no state to
/// reuse them across calls).
fn scatter_rows<T: DepthSample, A: Sync>(
    src: &[T],
    p: &RegisterParams,
    zbuf: &[A],
    u16_keys: bool,
    load: impl Fn(&A) -> u32 + Sync,
    store: impl Fn(&A, u32) + Sync,
    fetch_max: impl Fn(&A, u32) + Sync,
) {
    let w = p.src.width as usize;
    let sh = p.src.height as usize;
    if w == 0 || sh == 0 || p.width == 0 || p.height == 0 {
        return;
    }
    let BandLayout { offset, band_rows } = band_layout(p);
    let nb = sh.div_ceil(band_rows);
    let (nb_i, dh_i) = (nb as i64, i64::from(p.height));
    let lo = |i: usize| -> i64 {
        if i < 2 {
            i64::MIN
        } else {
            (2 * i as i64 - 1) * dh_i / (2 * nb_i) + offset
        }
    };
    let window = |i: usize| lo(i)..if i + 2 >= nb { i64::MAX } else { lo(i + 2) };
    let radius = i64::from(p.radius);
    let neon = cpu_features().has_neon;

    let band = |i: usize| -> Vec<Splat> {
        let win = window(i);
        let mut deferred = Vec::new();
        let mut batch = Batch::new();
        let y_end = ((i + 1) * band_rows).min(sh);
        for y in i * band_rows..y_end {
            let row = &src[y * w..(y + 1) * w];
            for (c, chunk) in row.chunks(PROJECT_BATCH).enumerate() {
                project_batch(chunk, c * PROJECT_BATCH, y, p, u16_keys, neon, &mut batch);
                let n = chunk.len();
                let cells = batch.key[..n].iter().zip(&batch.u[..n]).zip(&batch.v[..n]);
                for ((&key, &u), &v) in cells {
                    if key == 0 {
                        continue;
                    }
                    let v64 = i64::from(v);
                    let sp = Splat { u, v, key };
                    if v64 - radius >= win.start && v64 + radius < win.end {
                        splat(zbuf, p, sp, &load, &store);
                    } else {
                        deferred.push(sp);
                    }
                }
            }
        }
        deferred
    };

    // One buffer per band, kept as is (no merge copy).
    let mut deferred: Vec<Vec<Splat>> = Vec::new();
    for colour in 0..2 {
        deferred.par_extend(
            (0..nb.div_ceil(2))
                .into_par_iter()
                .map(|k| 2 * k + colour)
                .filter(|&i| i < nb)
                .map(band)
                .filter(|d| !d.is_empty()),
        );
    }
    // No plain store runs any more: the deferred splats go in parallel
    // through `fetch_max`.
    deferred
        .par_iter()
        .flat_map(|d| d.par_chunks(DEFERRED_CHUNK))
        .for_each(|c| c.iter().for_each(|&s| splat(zbuf, p, s, &load, &fetch_max)));
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
/// `!value` (so nearer = larger key, empty = `0` = the initial fill); the
/// finalize pass inverts the keys back.
pub(crate) fn register_u16(src: &[u16], dst: &mut [u16], p: &RegisterParams) {
    dst.par_chunks_mut(FILL_CHUNK).for_each(|c| c.fill(0));
    scatter_rows(
        src,
        p,
        as_atomic_u16(dst),
        true,
        |c| u32::from(c.load(Ordering::Relaxed)),
        // Keys of `key_u16` fit 16 bits.
        |c, k| c.store(k as u16, Ordering::Relaxed),
        |c, k| {
            c.fetch_max(k as u16, Ordering::Relaxed);
        },
    );
    dst.par_chunks_mut(FILL_CHUNK).for_each(|c| {
        for d in c {
            if *d != 0 {
                *d = !*d;
            }
        }
    });
}

/// Registers an `f32` depth image. The z-buffer is `dst`'s bit patterns,
/// holding `!bits(value)`: nearer = larger key and empty = `0` = the bits of
/// the `0.0` fill.
pub(crate) fn register_f32(src: &[f32], dst: &mut [f32], p: &RegisterParams) {
    dst.par_chunks_mut(FILL_CHUNK).for_each(|c| c.fill(0.0));
    scatter_rows(
        src,
        p,
        as_atomic_bits(dst),
        false,
        |c| c.load(Ordering::Relaxed),
        |c, k| c.store(k, Ordering::Relaxed),
        |c, k| {
            c.fetch_max(k, Ordering::Relaxed);
        },
    );
    dst.par_chunks_mut(FILL_CHUNK).for_each(|c| {
        for d in c {
            let k = d.to_bits();
            *d = f32::from_bits(if k != 0 { !k } else { 0 });
        }
    });
}
