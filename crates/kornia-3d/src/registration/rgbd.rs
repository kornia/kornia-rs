//! Depth-image (RGB-D) geometry for projective registration: vertex and normal maps, depth
//! halving, and the coarse-to-fine [`RgbdPyramid`].
//!
//! The surface measurement of KinectFusion (Newcombe et al., "KinectFusion: Real-time dense
//! surface mapping and tracking", ISMAR 2011, section 3.2): each depth pixel back-projected
//! through the pinhole intrinsics into a camera-frame vertex.
//!
//! Conventions:
//!
//! * Camera model: [`PinholeCamera`] without distortion (rectified depth), OpenCV convention,
//!   +Z forward, +X right, +Y down. Pixel centres sit at integer coordinates.
//! * Depth images: `Image<u16, 1>` in millimetres, `0` = invalid (no measurement), the rule of
//!   [`kornia_imgproc::depth::is_valid_depth_u16`]; or `Image<f32, 1>` in metres, valid iff
//!   finite and `> 0`, the rule of [`kornia_imgproc::depth::is_valid_depth_f32`].
//! * Depth sigma maps: `Image<f32, 1>` of per-pixel one-sigma depth (`z`) errors in metres, the
//!   size of their depth; `0` or non-finite = no value for that pixel.
//! * Vertex maps: `Image<f32, 3>` of camera-frame points in metres, `[0, 0, 0]` = invalid
//!   (a valid vertex always has `z > 0`).
//! * Normal maps: `Image<f32, 3>` of unit normals oriented towards the camera (`n . v < 0`),
//!   `[0, 0, 0]` = invalid.

use crate::camera::PinholeCamera;
use kornia_algebra::Vec3F32;
use kornia_image::{Image, ImageError, ImageSize};
use kornia_imgproc::depth::{is_valid_depth_f32, is_valid_depth_u16};

/// Errors produced by the RGB-D depth-geometry routines.
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum RgbdIcpError {
    /// The camera has non-zero distortion; these routines need rectified depth.
    #[error("camera has non-zero distortion; depth must be rectified")]
    DistortedCamera,
    /// The focal lengths are not finite and positive, or the principal point is not finite.
    #[error("invalid intrinsics: fx {fx}, fy {fy}, cx {cx}, cy {cy}")]
    InvalidIntrinsics {
        /// Focal length along x, in pixels.
        fx: f64,
        /// Focal length along y, in pixels.
        fy: f64,
        /// Principal point x, in pixels.
        cx: f64,
        /// Principal point y, in pixels.
        cy: f64,
    },
    /// An output image does not have the size its input requires.
    #[error("output image is {got:?}, expected {expected:?}")]
    SizeMismatch {
        /// Size the output must have.
        expected: ImageSize,
        /// Size of the given output.
        got: ImageSize,
    },
    /// A pyramid was requested with zero levels.
    #[error("pyramid needs at least 1 level, got {0}")]
    InvalidNumLevels(usize),
    /// The image becomes too small to process at the requested pyramid level.
    #[error("image is {width}x{height} at pyramid level {level}; need at least 3x3")]
    ImageTooSmall {
        /// Pyramid level at which the image became too small (0 = finest).
        level: usize,
        /// Image width at that level.
        width: usize,
        /// Image height at that level.
        height: usize,
    },
    /// An image operation failed.
    #[error(transparent)]
    Image(#[from] ImageError),
    /// The source and target pyramids have different numbers of levels.
    #[error("source pyramid has {src_levels} levels, target has {tgt_levels}")]
    PyramidLevelsMismatch {
        /// Levels of the source pyramid.
        src_levels: usize,
        /// Levels of the target pyramid.
        tgt_levels: usize,
    },
    /// The ICP iteration budget does not have one entry per pyramid level.
    #[error("iters_per_level has {iters} entries for {levels} pyramid levels")]
    IterationsLevelsMismatch {
        /// Entries of `iters_per_level`.
        iters: usize,
        /// Levels of the pyramids.
        levels: usize,
    },
    /// An ICP threshold is out of its range.
    #[error("invalid ICP criteria: {field} = {value}")]
    InvalidCriteria {
        /// Name of the offending field.
        field: &'static str,
        /// Its value.
        value: f64,
    },
    /// Too few correspondences passed the gates to constrain a pose: low overlap or a bad
    /// initial guess.
    #[error("only {0} correspondences passed the gates; need at least 6")]
    TooFewCorrespondences(usize),
    /// The point-to-plane normal equations or their solve are not finite. Degenerate geometry,
    /// such as a single plane, is not an error: it is solved along its observed directions.
    #[error("point-to-plane normal equations or their solve are not finite")]
    SingularNormalEquations,
}

/// Returns `Ok` if `camera` has finite positive focal lengths, a finite principal point and no
/// distortion, so that a pixel and its ray relate through `fx, fy, cx, cy` alone and every
/// vertex computed from valid depth is finite.
fn check_camera(camera: &PinholeCamera) -> Result<(), RgbdIcpError> {
    let (fx, fy, cx, cy) = camera.intrinsics();
    // written as pass tests so that NaN fails them
    let focal_ok = fx > 0.0 && fy > 0.0 && fx.is_finite() && fy.is_finite();
    let centre_ok = cx.is_finite() && cy.is_finite();
    if !(focal_ok && centre_ok) {
        return Err(RgbdIcpError::InvalidIntrinsics { fx, fy, cx, cy });
    }
    if [camera.k1, camera.k2, camera.p1, camera.p2]
        .iter()
        .all(|&k| k == 0.0)
    {
        Ok(())
    } else {
        Err(RgbdIcpError::DistortedCamera)
    }
}

/// Returns `Ok` if `got` equals `expected`.
fn check_size(expected: ImageSize, got: ImageSize) -> Result<(), RgbdIcpError> {
    if expected == got {
        Ok(())
    } else {
        Err(RgbdIcpError::SizeMismatch { expected, got })
    }
}

/// Returns true if a vertex-map entry is a valid measurement: finite, with `z > 0`.
#[inline]
pub fn is_valid_vertex(v: &[f32; 3]) -> bool {
    v[2] > 0.0 && v.iter().all(|c| c.is_finite())
}

/// Back-projects a millimetre depth image into a camera-frame vertex map in metres.
///
/// Pixel `(u, v)` with depth `d > 0` maps to `z * [(u - cx) / fx, (v - cy) / fy, 1]` with
/// `z = d / 1000`. Invalid depth (`0`) maps to `[0, 0, 0]`. The inverse of
/// [`PinholeCamera::project_to_pixel`] for an undistorted camera.
///
/// # Arguments
///
/// * `src` - Depth in millimetres, `0` = invalid.
/// * `camera` - Intrinsics of `src`, without distortion.
/// * `dst` - Output vertex map, the size of `src`.
///
/// # Returns
///
/// `Ok(())` once `dst` holds the vertex map.
///
/// # Errors
///
/// Returns [`RgbdIcpError::DistortedCamera`] if `camera` has non-zero distortion,
/// [`RgbdIcpError::InvalidIntrinsics`] if its focal lengths are not finite and positive or its
/// principal point is not finite, and
/// [`RgbdIcpError::SizeMismatch`] if `dst` is not the size of `src`.
///
/// # Example
///
/// ```
/// use kornia_3d::camera::PinholeCamera;
/// use kornia_3d::registration::vertex_map_from_depth_mm;
/// use kornia_image::{Image, ImageSize};
///
/// let size = ImageSize { width: 4, height: 3 };
/// let depth = Image::<u16, 1>::from_size_val(size, 2000)?;
/// let camera = PinholeCamera { fx: 2.0, fy: 2.0, cx: 1.5, cy: 1.0, ..PinholeCamera::IDENTITY };
/// let mut vertices = Image::<f32, 3>::from_size_val(size, 0.0)?;
/// vertex_map_from_depth_mm(&depth, &camera, &mut vertices)?;
/// // pixel (0, 0) at 2 m: x = (0 - 1.5) * 2 / 2, y = (0 - 1) * 2 / 2
/// assert_eq!(&vertices.as_slice()[..3], &[-1.5, -1.0, 2.0]);
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
pub fn vertex_map_from_depth_mm(
    src: &Image<u16, 1>,
    camera: &PinholeCamera,
    dst: &mut Image<f32, 3>,
) -> Result<(), RgbdIcpError> {
    vertex_map_from_depth(src, camera, dst, |d| {
        is_valid_depth_u16(d).then_some(d as f64 * 1e-3)
    })
}

/// Back-projects a metre depth image into a camera-frame vertex map in metres.
///
/// As [`vertex_map_from_depth_mm`] without the millimetre quantisation: pixel `(u, v)` with
/// valid depth `z` maps to `z * [(u - cx) / fx, (v - cy) / fy, 1]`. Depth that is `0`,
/// negative or not finite (see [`kornia_imgproc::depth::is_valid_depth_f32`]) maps to
/// `[0, 0, 0]`.
///
/// # Arguments
///
/// * `src` - Depth in metres; `0`, negative or non-finite = invalid.
/// * `camera` - Intrinsics of `src`, without distortion.
/// * `dst` - Output vertex map, the size of `src`.
///
/// # Returns
///
/// `Ok(())` once `dst` holds the vertex map.
///
/// # Errors
///
/// As [`vertex_map_from_depth_mm`]: [`RgbdIcpError::DistortedCamera`],
/// [`RgbdIcpError::InvalidIntrinsics`] or [`RgbdIcpError::SizeMismatch`].
pub fn vertex_map_from_depth_m(
    src: &Image<f32, 1>,
    camera: &PinholeCamera,
    dst: &mut Image<f32, 3>,
) -> Result<(), RgbdIcpError> {
    vertex_map_from_depth(src, camera, dst, |d| {
        is_valid_depth_f32(d).then_some(d as f64)
    })
}

/// Back-projection shared by the depth types; `z_m` returns a sample's depth in metres, `None`
/// when invalid. Monomorphised per type, so the `u16` path keeps its integer test.
fn vertex_map_from_depth<T: Copy>(
    src: &Image<T, 1>,
    camera: &PinholeCamera,
    dst: &mut Image<f32, 3>,
    z_m: impl Fn(T) -> Option<f64>,
) -> Result<(), RgbdIcpError> {
    check_camera(camera)?;
    check_size(src.size(), dst.size())?;

    let width = src.width();
    let (fx, fy, cx, cy) = camera.intrinsics();
    let rows = src.as_slice().chunks_exact(width.max(1));
    let out_rows = dst.as_slice_mut().chunks_exact_mut(3 * width.max(1));
    for (v, (row, out_row)) in rows.zip(out_rows).enumerate() {
        for (u, (&d, out_px)) in row.iter().zip(out_row.chunks_exact_mut(3)).enumerate() {
            let Some(z) = z_m(d) else {
                out_px.fill(0.0);
                continue;
            };
            out_px[0] = ((u as f64 - cx) * z / fx) as f32;
            out_px[1] = ((v as f64 - cy) * z / fy) as f32;
            out_px[2] = z as f32;
        }
    }

    Ok(())
}

/// Returns true if a normal-map entry is valid: finite and non-zero.
#[inline]
pub fn is_valid_normal(n: &[f32; 3]) -> bool {
    (n[0] != 0.0 || n[1] != 0.0 || n[2] != 0.0) && n.iter().all(|c| c.is_finite())
}

/// Computes a normal map from a vertex map by the cross product of central differences.
///
/// For each interior pixel with a valid centre and four valid neighbours, the normal is
/// `normalize((v[u+1] - v[u-1]) x (v[v+1] - v[v-1]))`, flipped to face the camera
/// (`n . vertex < 0`). Border pixels and pixels with an invalid neighbour get `[0, 0, 0]`.
/// This is the normal estimate of KinectFusion (Newcombe et al., ISMAR 2011, eq. 2), with
/// central instead of forward differences, so the normal is centred on its pixel.
///
/// # Arguments
///
/// * `src` - Camera-frame vertex map, `[0, 0, 0]` = invalid.
/// * `dst` - Output normal map, the size of `src`.
///
/// # Returns
///
/// `Ok(())` once `dst` holds the normal map.
///
/// # Errors
///
/// Returns [`RgbdIcpError::SizeMismatch`] if `dst` is not the size of `src`.
pub fn normal_map_from_vertex_map(
    src: &Image<f32, 3>,
    dst: &mut Image<f32, 3>,
) -> Result<(), RgbdIcpError> {
    check_size(src.size(), dst.size())?;
    let (width, height) = (src.width(), src.height());
    let vertex = |idx: usize| -> [f32; 3] {
        let s = &src.as_slice()[3 * idx..3 * idx + 3];
        [s[0], s[1], s[2]]
    };
    let out = dst.as_slice_mut();
    out.fill(0.0);
    if width < 3 || height < 3 {
        return Ok(());
    }

    for y in 1..height - 1 {
        for x in 1..width - 1 {
            let idx = y * width + x;
            let center = vertex(idx);
            let (left, right) = (vertex(idx - 1), vertex(idx + 1));
            let (up, down) = (vertex(idx - width), vertex(idx + width));
            if ![center, left, right, up, down].iter().all(is_valid_vertex) {
                continue;
            }

            let v = Vec3F32::from_array;
            // central differences along u and v; their cross product is the surface normal
            let n = (v(right) - v(left)).cross(v(down) - v(up));
            let norm = n.length();
            if norm.is_nan() || norm < 1e-12 {
                continue;
            }

            // orient towards the camera: the view ray is the vertex itself
            let flip = if n.dot(v(center)) > 0.0 { -1.0 } else { 1.0 };
            out[3 * idx..3 * idx + 3].copy_from_slice(&(n / norm * flip).to_array());
        }
    }

    Ok(())
}

/// Largest in-block depth spread that is still averaged together when downsampling; samples
/// further than this from the block median belong to another surface (a depth
/// discontinuity) and are excluded.
const DOWNSAMPLE_JUMP_MM: i32 = 50;

/// Halves a millimetre depth image, validity- and discontinuity-aware.
///
/// Each output pixel covers a 2x2 input block. Invalid (`0`) samples are ignored, the lower
/// median of the valid samples is the pivot, and only samples within 50 mm of the pivot are
/// averaged (rounded). So the output never averages across a depth discontinuity and never
/// fills holes: a fully invalid block stays `0`. A trailing odd row or column is dropped.
/// KinectFusion builds its depth pyramid the same way (block average within `3 sigma` of the
/// centre); the median pivot here keeps a single foreground sample from choosing the surface.
///
/// # Arguments
///
/// * `src` - Depth in millimetres, `0` = invalid.
/// * `dst` - Output depth of size `(width / 2, height / 2)` (floor).
///
/// # Returns
///
/// `Ok(())` once `dst` holds the halved depth.
///
/// # Errors
///
/// Returns [`RgbdIcpError::SizeMismatch`] if `dst` is not `(width / 2, height / 2)`.
pub fn downsample_depth_half(
    src: &Image<u16, 1>,
    dst: &mut Image<u16, 1>,
) -> Result<(), RgbdIcpError> {
    let half = ImageSize {
        width: src.width() / 2,
        height: src.height() / 2,
    };
    check_size(half, dst.size())?;
    let (width, depth) = (src.width(), src.as_slice());

    for (y, out_row) in dst
        .as_slice_mut()
        .chunks_exact_mut(half.width.max(1))
        .enumerate()
    {
        let row0 = &depth[2 * y * width..];
        let row1 = &depth[(2 * y + 1) * width..];
        for (x, out) in out_row.iter_mut().enumerate() {
            let block = [row0[2 * x], row0[2 * x + 1], row1[2 * x], row1[2 * x + 1]];
            let mut valid = [0u16; 4];
            let mut n = 0;
            for &d in &block {
                if is_valid_depth_u16(d) {
                    valid[n] = d;
                    n += 1;
                }
            }
            if n == 0 {
                *out = 0;
                continue;
            }
            valid[..n].sort_unstable();
            // lower median: always an actual sample, so it cannot itself straddle a
            // discontinuity
            let pivot = valid[(n - 1) / 2] as i32;
            let (mut sum, mut count) = (0u32, 0u32);
            for &d in &valid[..n] {
                if (d as i32 - pivot).abs() <= DOWNSAMPLE_JUMP_MM {
                    sum += d as u32;
                    count += 1;
                }
            }
            *out = ((sum + count / 2) / count) as u16;
        }
    }

    Ok(())
}

/// [`DOWNSAMPLE_JUMP_MM`] in metres, for the `f32` path.
const DOWNSAMPLE_JUMP_M: f32 = DOWNSAMPLE_JUMP_MM as f32 * 1e-3;

/// Halves a metre depth image, validity- and discontinuity-aware.
///
/// The rule of [`downsample_depth_half`] on `f32` metres: the lower median of the valid samples
/// of each 2x2 block is the pivot and the samples within 50 mm of it are averaged, without
/// rounding. A block with no valid sample (see [`kornia_imgproc::depth::is_valid_depth_f32`])
/// becomes `0`. A trailing odd row or column is dropped.
///
/// # Arguments
///
/// * `src` - Depth in metres; `0`, negative or non-finite = invalid.
/// * `dst` - Output depth of size `(width / 2, height / 2)` (floor).
///
/// # Returns
///
/// `Ok(())` once `dst` holds the halved depth.
///
/// # Errors
///
/// Returns [`RgbdIcpError::SizeMismatch`] if `dst` is not `(width / 2, height / 2)`.
pub fn downsample_depth_half_m(
    src: &Image<f32, 1>,
    dst: &mut Image<f32, 1>,
) -> Result<(), RgbdIcpError> {
    downsample_half_m(src, dst, None)
}

/// Halves a metre depth image as [`downsample_depth_half_m`] and, with `sigmas = (src, dst)`,
/// its sigma map alongside. The averaged depth is the mean of `n` independent samples, so its
/// sigma is `sqrt(sum sigma_k^2) / n`; `0` (no value) when an averaged sample has none.
fn downsample_half_m(
    src: &Image<f32, 1>,
    dst: &mut Image<f32, 1>,
    mut sigmas: Option<(&Image<f32, 1>, &mut Image<f32, 1>)>,
) -> Result<(), RgbdIcpError> {
    let half = ImageSize {
        width: src.width() / 2,
        height: src.height() / 2,
    };
    check_size(half, dst.size())?;
    if let Some((s_src, s_dst)) = &sigmas {
        check_size(src.size(), s_src.size())?;
        check_size(half, s_dst.size())?;
    }
    let (width, depth) = (src.width(), src.as_slice());

    for (y, out_row) in dst
        .as_slice_mut()
        .chunks_exact_mut(half.width.max(1))
        .enumerate()
    {
        for (x, out) in out_row.iter_mut().enumerate() {
            let (i0, i1) = (2 * y * width + 2 * x, (2 * y + 1) * width + 2 * x);
            let mut valid = [(0.0f32, 0usize); 4];
            let mut n = 0;
            for i in [i0, i0 + 1, i1, i1 + 1] {
                if is_valid_depth_f32(depth[i]) {
                    valid[n] = (depth[i], i);
                    n += 1;
                }
            }
            let out_idx = y * half.width + x;
            if n == 0 {
                *out = 0.0;
                if let Some((_, s_dst)) = sigmas.as_mut() {
                    s_dst.as_slice_mut()[out_idx] = 0.0;
                }
                continue;
            }
            valid[..n].sort_unstable_by(|a, b| a.0.total_cmp(&b.0));
            // lower median, as on the millimetre path
            let pivot = valid[(n - 1) / 2].0;
            let (mut sum, mut var, mut count, mut sigma_ok) = (0.0f64, 0.0f64, 0u32, true);
            for &(d, i) in &valid[..n] {
                if (d - pivot).abs() <= DOWNSAMPLE_JUMP_M {
                    sum += d as f64;
                    count += 1;
                    if let Some((s_src, _)) = &sigmas {
                        let s = s_src.as_slice()[i];
                        // written so that NaN fails
                        sigma_ok &= s > 0.0 && s.is_finite();
                        var += s as f64 * s as f64;
                    }
                }
            }
            *out = (sum / count as f64) as f32;
            if let Some((_, s_dst)) = sigmas.as_mut() {
                s_dst.as_slice_mut()[out_idx] = if sigma_ok {
                    (var.sqrt() / count as f64) as f32
                } else {
                    0.0
                };
            }
        }
    }

    Ok(())
}

/// Intrinsics of the image [`downsample_depth_half`] produces from one seen by `camera`.
///
/// With pixel centres at integer coordinates, a half-resolution pixel `u` covers the 2x2
/// full-resolution block centred at `2u + 0.5`, so `u_full = 2 u_half + 0.5`. Substituting into
/// the projection gives `fx' = fx / 2` and `cx' = (cx + 0.5) / 2 - 0.5` (likewise for y).
fn half_camera(camera: &PinholeCamera) -> PinholeCamera {
    PinholeCamera {
        fx: camera.fx / 2.0,
        fy: camera.fy / 2.0,
        cx: (camera.cx + 0.5) / 2.0 - 0.5,
        cy: (camera.cy + 0.5) / 2.0 - 0.5,
        ..camera.clone()
    }
}

/// One resolution level of an [`RgbdPyramid`]: intrinsics plus vertex and normal maps.
///
/// Built by [`RgbdPyramid`] only; later per-level maps become new fields.
#[derive(Clone)]
#[non_exhaustive]
pub struct RgbdLevel {
    /// Intrinsics at this level's resolution, without distortion.
    pub(crate) camera: PinholeCamera,
    /// Camera-frame vertex map in metres, `[0, 0, 0]` = invalid.
    pub(crate) vertices: Image<f32, 3>,
    /// Unit normal map oriented towards the camera, `[0, 0, 0]` = invalid.
    pub(crate) normals: Image<f32, 3>,
    /// Per-pixel one-sigma depth error in metres, the size of the maps; `None` when the frame
    /// carries none.
    pub(crate) depth_sigmas: Option<Image<f32, 1>>,
}

// Fields stay crate-private so the maps always share one size and match `camera`: the
// association indexes all of them with one pixel index.
impl RgbdLevel {
    /// Size of this level's maps in pixels.
    pub fn size(&self) -> ImageSize {
        self.vertices.size()
    }

    /// Intrinsics at this level's resolution, without distortion.
    pub fn camera(&self) -> &PinholeCamera {
        &self.camera
    }

    /// Camera-frame vertex map in metres, `[0, 0, 0]` = invalid.
    pub fn vertices(&self) -> &Image<f32, 3> {
        &self.vertices
    }

    /// Unit normal map oriented towards the camera, `[0, 0, 0]` = invalid.
    pub fn normals(&self) -> &Image<f32, 3> {
        &self.normals
    }

    /// Per-pixel one-sigma depth (`z`) error in metres, `None` when the frame carries none.
    ///
    /// Where present it takes precedence over [`super::IcpPlaneCriteria::depth_noise`] for this
    /// frame's points; `0` or non-finite entries fall back to that model. Coarser levels hold
    /// the sigma of their averaged depth (see [`downsample_depth_half_m`]).
    pub fn depth_sigmas(&self) -> Option<&Image<f32, 1>> {
        self.depth_sigmas.as_ref()
    }
}

/// Coarse-to-fine pyramid of vertex and normal maps built from one depth image.
///
/// `levels[0]` is the input resolution; each further level halves the depth with
/// [`downsample_depth_half`] (or [`downsample_depth_half_m`]) and the intrinsics to match.
#[derive(Clone)]
pub struct RgbdPyramid {
    /// Pyramid levels, finest first.
    pub(crate) levels: Vec<RgbdLevel>,
}

impl RgbdPyramid {
    /// Pyramid levels, finest first.
    pub fn levels(&self) -> &[RgbdLevel] {
        &self.levels
    }

    /// Builds a pyramid with `num_levels` levels from a millimetre depth image.
    ///
    /// # Arguments
    ///
    /// * `depth` - Depth in millimetres, `0` = invalid.
    /// * `camera` - Intrinsics of `depth`, without distortion.
    /// * `num_levels` - Number of levels, at least 1.
    ///
    /// # Returns
    ///
    /// The pyramid, finest level first.
    ///
    /// # Errors
    ///
    /// Returns [`RgbdIcpError::InvalidNumLevels`] if `num_levels` is 0,
    /// [`RgbdIcpError::ImageTooSmall`] if a level would be smaller than 3x3, and
    /// [`RgbdIcpError::DistortedCamera`] if `camera` has non-zero distortion, and
    /// [`RgbdIcpError::InvalidIntrinsics`] if its focal lengths are not finite and positive or
    /// its principal point is not finite.
    ///
    /// # Example
    ///
    /// ```
    /// use kornia_3d::camera::PinholeCamera;
    /// use kornia_3d::registration::RgbdPyramid;
    /// use kornia_image::{Image, ImageSize};
    ///
    /// let size = ImageSize { width: 64, height: 48 };
    /// let depth = Image::<u16, 1>::from_size_val(size, 1500)?;
    /// let camera = PinholeCamera { fx: 50.0, fy: 50.0, cx: 31.5, cy: 23.5, ..PinholeCamera::IDENTITY };
    /// let pyramid = RgbdPyramid::from_depth_mm(&depth, &camera, 3)?;
    /// assert_eq!(pyramid.levels()[2].size(), ImageSize { width: 16, height: 12 });
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    pub fn from_depth_mm(
        depth: &Image<u16, 1>,
        camera: &PinholeCamera,
        num_levels: usize,
    ) -> Result<Self, RgbdIcpError> {
        Self::build(
            depth,
            None,
            camera,
            num_levels,
            0u16,
            |src, dst, _| downsample_depth_half(src, dst),
            vertex_map_from_depth_mm,
        )
    }

    /// Builds a pyramid with `num_levels` levels from a metre depth image.
    ///
    /// As [`Self::from_depth_mm`] without the millimetre quantisation; coarser levels halve the
    /// depth with [`downsample_depth_half_m`].
    ///
    /// # Arguments
    ///
    /// * `depth` - Depth in metres; `0`, negative or non-finite = invalid.
    /// * `camera` - Intrinsics of `depth`, without distortion.
    /// * `num_levels` - Number of levels, at least 1.
    ///
    /// # Returns
    ///
    /// The pyramid, finest level first, without depth sigma maps.
    ///
    /// # Errors
    ///
    /// As [`Self::from_depth_mm`].
    ///
    /// # Example
    ///
    /// ```
    /// use kornia_3d::camera::PinholeCamera;
    /// use kornia_3d::registration::RgbdPyramid;
    /// use kornia_image::{Image, ImageSize};
    ///
    /// let size = ImageSize { width: 64, height: 48 };
    /// let depth = Image::<f32, 1>::from_size_val(size, 1.5)?;
    /// let camera = PinholeCamera { fx: 50.0, fy: 50.0, cx: 31.5, cy: 23.5, ..PinholeCamera::IDENTITY };
    /// let pyramid = RgbdPyramid::from_depth_m(&depth, &camera, 2)?;
    /// assert!(pyramid.levels()[1].depth_sigmas().is_none());
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    pub fn from_depth_m(
        depth: &Image<f32, 1>,
        camera: &PinholeCamera,
        num_levels: usize,
    ) -> Result<Self, RgbdIcpError> {
        Self::build(
            depth,
            None,
            camera,
            num_levels,
            0.0f32,
            downsample_half_m,
            vertex_map_from_depth_m,
        )
    }

    /// Builds a pyramid as [`Self::from_depth_m`], carrying a per-pixel depth sigma map into
    /// every level ([`RgbdLevel::depth_sigmas`]).
    ///
    /// # Arguments
    ///
    /// * `depth` - Depth in metres; `0`, negative or non-finite = invalid.
    /// * `sigma` - One-sigma depth error in metres per pixel, the size of `depth`; `0` or
    ///   non-finite = no value for that pixel.
    /// * `camera` - Intrinsics of `depth`, without distortion.
    /// * `num_levels` - Number of levels, at least 1.
    ///
    /// # Returns
    ///
    /// The pyramid, finest level first, each level with its sigma map.
    ///
    /// # Errors
    ///
    /// As [`Self::from_depth_mm`], and [`RgbdIcpError::SizeMismatch`] if `sigma` is not the size
    /// of `depth`.
    pub fn from_depth_m_with_sigma(
        depth: &Image<f32, 1>,
        sigma: &Image<f32, 1>,
        camera: &PinholeCamera,
        num_levels: usize,
    ) -> Result<Self, RgbdIcpError> {
        check_size(depth.size(), sigma.size())?;
        Self::build(
            depth,
            Some(sigma),
            camera,
            num_levels,
            0.0f32,
            downsample_half_m,
            vertex_map_from_depth_m,
        )
    }

    /// The level loop shared by the depth types: `halve` downsamples a level's depth (and its
    /// sigma map, when given) into the next, `unproject` turns a depth into a vertex map.
    fn build<T: Copy>(
        depth: &Image<T, 1>,
        sigma: Option<&Image<f32, 1>>,
        camera: &PinholeCamera,
        num_levels: usize,
        invalid: T,
        halve: impl Fn(
            &Image<T, 1>,
            &mut Image<T, 1>,
            Option<(&Image<f32, 1>, &mut Image<f32, 1>)>,
        ) -> Result<(), RgbdIcpError>,
        unproject: impl Fn(&Image<T, 1>, &PinholeCamera, &mut Image<f32, 3>) -> Result<(), RgbdIcpError>,
    ) -> Result<Self, RgbdIcpError> {
        if num_levels == 0 {
            return Err(RgbdIcpError::InvalidNumLevels(num_levels));
        }
        check_camera(camera)?;

        let mut levels = Vec::with_capacity(num_levels);
        let mut level_camera = camera.clone();
        // level 0 reads `depth` and `sigma` directly; `halved` holds them from level 1 on
        let mut halved: Option<(Image<T, 1>, Option<Image<f32, 1>>)> = None;
        for level in 0..num_levels {
            let (cur, cur_sigma) = match &halved {
                Some((d, s)) => (d, s.as_ref()),
                None => (depth, sigma),
            };
            let size = if level == 0 {
                cur.size()
            } else {
                ImageSize {
                    width: cur.width() / 2,
                    height: cur.height() / 2,
                }
            };
            if size.width < 3 || size.height < 3 {
                return Err(RgbdIcpError::ImageTooSmall {
                    level,
                    width: size.width,
                    height: size.height,
                });
            }
            if level > 0 {
                let mut half = Image::from_size_val(size, invalid)?;
                let mut half_sigma = match cur_sigma {
                    Some(_) => Some(Image::from_size_val(size, 0.0f32)?),
                    None => None,
                };
                halve(cur, &mut half, cur_sigma.zip(half_sigma.as_mut()))?;
                halved = Some((half, half_sigma));
                level_camera = half_camera(&level_camera);
            }
            let (cur, cur_sigma) = match &halved {
                Some((d, s)) => (d, s.as_ref()),
                None => (depth, sigma),
            };
            let mut vertices = Image::from_size_val(size, 0.0f32)?;
            unproject(cur, &level_camera, &mut vertices)?;
            let mut normals = Image::from_size_val(size, 0.0f32)?;
            normal_map_from_vertex_map(&vertices, &mut normals)?;
            levels.push(RgbdLevel {
                camera: level_camera.clone(),
                vertices,
                normals,
                depth_sigmas: cur_sigma.cloned(),
            });
        }

        Ok(Self { levels })
    }

    /// Rebuilds this pyramid in place from a millimetre depth image, reusing its maps.
    ///
    /// The same pyramid as [`Self::from_depth_mm`], bit for bit. A level whose size is
    /// unchanged keeps its images, and `scratch` keeps the halved depths, so once this pyramid
    /// and `scratch` have seen an image size a call at that size allocates nothing. The input
    /// is checked before anything is written, so an invalid one leaves the pyramid unchanged.
    ///
    /// # Arguments
    ///
    /// * `depth` - Depth in millimetres, `0` = invalid.
    /// * `camera` - Intrinsics of `depth`, without distortion.
    /// * `num_levels` - Number of levels, at least 1.
    /// * `scratch` - Halved depths reused across calls.
    ///
    /// # Returns
    ///
    /// `Ok(())` once the pyramid holds the levels of `depth`.
    ///
    /// # Errors
    ///
    /// As [`Self::from_depth_mm`].
    ///
    /// # Example
    ///
    /// ```
    /// use kornia_3d::camera::PinholeCamera;
    /// use kornia_3d::registration::{PyramidScratch, RgbdPyramid};
    /// use kornia_image::{Image, ImageSize};
    ///
    /// let size = ImageSize { width: 64, height: 48 };
    /// let camera = PinholeCamera { fx: 50.0, fy: 50.0, cx: 31.5, cy: 23.5, ..PinholeCamera::IDENTITY };
    /// let mut pyramid = RgbdPyramid::from_depth_mm(&Image::from_size_val(size, 1500)?, &camera, 3)?;
    /// let mut scratch = PyramidScratch::default();
    /// pyramid.rebuild_from_depth_mm(&Image::from_size_val(size, 1600)?, &camera, 3, &mut scratch)?;
    /// assert_eq!(pyramid.levels()[0].vertices().as_slice()[2], 1.6);
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    pub fn rebuild_from_depth_mm(
        &mut self,
        depth: &Image<u16, 1>,
        camera: &PinholeCamera,
        num_levels: usize,
        scratch: &mut PyramidScratch,
    ) -> Result<(), RgbdIcpError> {
        if num_levels == 0 {
            return Err(RgbdIcpError::InvalidNumLevels(num_levels));
        }
        check_camera(camera)?;
        // repeated floor halving is a floor division by 2^level
        let level_size = |level: usize| ImageSize {
            width: depth.width() >> level,
            height: depth.height() >> level,
        };
        for level in 0..num_levels {
            let size = level_size(level);
            if size.width < 3 || size.height < 3 {
                return Err(RgbdIcpError::ImageTooSmall {
                    level,
                    width: size.width,
                    height: size.height,
                });
            }
        }

        // every pixel of a halved depth and of a vertex map is written, and the normal map is
        // cleared first, so a reused image carries nothing of the frame before
        let halved = &mut scratch.halved;
        halved.truncate(num_levels - 1);
        for level in 1..num_levels {
            let size = level_size(level);
            match halved.get(level - 1) {
                Some(h) if h.size() == size => {}
                Some(_) => halved[level - 1] = Image::from_size_val(size, 0)?,
                None => halved.push(Image::from_size_val(size, 0)?),
            }
            let (done, rest) = halved.split_at_mut(level - 1);
            let src = done.last().unwrap_or(depth);
            downsample_depth_half(src, &mut rest[0])?;
        }

        self.levels.truncate(num_levels);
        let mut level_camera = camera.clone();
        for level in 0..num_levels {
            if level > 0 {
                level_camera = half_camera(&level_camera);
            }
            let size = level_size(level);
            if self.levels.get(level).is_none_or(|l| l.size() != size) {
                let blank = RgbdLevel {
                    camera: level_camera.clone(),
                    vertices: Image::from_size_val(size, 0.0)?,
                    normals: Image::from_size_val(size, 0.0)?,
                    depth_sigmas: None,
                };
                match self.levels.get_mut(level) {
                    Some(l) => *l = blank,
                    None => self.levels.push(blank),
                }
            }
            let cur = if level == 0 {
                depth
            } else {
                &halved[level - 1]
            };
            let out = &mut self.levels[level];
            out.camera = level_camera.clone();
            out.depth_sigmas = None;
            vertex_map_from_depth_mm(cur, &level_camera, &mut out.vertices)?;
            normal_map_from_vertex_map(&out.vertices, &mut out.normals)?;
        }
        Ok(())
    }
}

/// Reused halved depths for [`RgbdPyramid::rebuild_from_depth_mm`], one per level below the
/// finest.
#[derive(Clone, Default)]
pub struct PyramidScratch {
    halved: Vec<Image<u16, 1>>,
}

#[cfg(test)]
mod tests {
    use super::super::synth::{
        depth_m_from_mm, render_depth_mm, render_vertex_map, test_camera, Plane, Scene, TEST_SIZE,
    };
    use super::*;
    use kornia_algebra::{Mat3F64, Vec3F64};

    #[test]
    fn test_vertex_map_round_trip() -> Result<(), Box<dyn std::error::Error>> {
        let camera = test_camera();
        let depth = render_depth_mm(
            &Scene::corner_and_sphere(),
            &camera,
            TEST_SIZE,
            &Mat3F64::IDENTITY,
            &Vec3F64::ZERO,
        )?;

        let mut verts = Image::<f32, 3>::from_size_val(TEST_SIZE, 0.0)?;
        vertex_map_from_depth_mm(&depth, &camera, &mut verts)?;

        let mut checked = 0;
        let pixels = depth
            .as_slice()
            .iter()
            .zip(verts.as_slice().chunks_exact(3));
        for (idx, (&d, p)) in pixels.enumerate() {
            let (u, v) = (idx % TEST_SIZE.width, idx / TEST_SIZE.width);
            if d == 0 {
                assert_eq!(p, [0.0; 3], "invalid depth must give [0,0,0] at ({u},{v})");
                continue;
            }
            assert!(is_valid_vertex(&[p[0], p[1], p[2]]));
            // z is the depth in metres
            assert!((p[2] as f64 - d as f64 * 1e-3).abs() < 1e-6);
            // project(unproject(u, v, z)) == (u, v)
            let uv = camera
                .project_to_pixel(&Vec3F64::new(p[0] as f64, p[1] as f64, p[2] as f64), 0.0)
                .ok_or("valid vertex must project")?;
            assert!(
                (uv.x - u as f64).abs() < 1e-3 && (uv.y - v as f64).abs() < 1e-3,
                "round trip failed at ({u},{v}): got ({}, {})",
                uv.x,
                uv.y
            );
            checked += 1;
        }
        assert!(
            checked > 10_000,
            "scene should cover most pixels: {checked}"
        );
        Ok(())
    }

    /// Values from numpy: `z = d * 1e-3; x = (u - cx) * z / fx; y = (v - cy) * z / fy`, in
    /// `float64` then cast to `float32`, with `d == 0` set to zero. Covers an invalid pixel,
    /// the smallest and largest `u16` depths and a principal point off the pixel grid.
    #[test]
    fn test_vertex_map_matches_numpy_reference() -> Result<(), Box<dyn std::error::Error>> {
        let size = ImageSize {
            width: 3,
            height: 2,
        };
        let depth = Image::<u16, 1>::new(size, vec![1000, 0, 2375, 65535, 1, 40000])?;
        let camera = PinholeCamera {
            fx: 525.0,
            fy: 519.5,
            cx: 1.25,
            cy: 0.75,
            ..PinholeCamera::IDENTITY
        };
        #[rustfmt::skip]
        let expected: [f64; 18] = [
            -0.0023809524718672037, -0.0014436958590522408, 1.0,
            0.0, 0.0, 0.0,
            0.0033928570337593555, -0.0034287776798009872, 2.375,
            -0.15603572130203247, 0.03153753653168678, 65.53500366210938,
            -4.76190479048455e-07, 4.812319502889295e-07, 0.0010000000474974513,
            0.05714285746216774, 0.01924927905201912, 40.0,
        ];
        let mut verts = Image::<f32, 3>::from_size_val(size, 7.0)?;
        vertex_map_from_depth_mm(&depth, &camera, &mut verts)?;
        for (i, (&got, &want)) in verts.as_slice().iter().zip(&expected).enumerate() {
            // both sides round the same f64 value to f32
            assert_eq!(got, want as f32, "component {i}");
        }
        Ok(())
    }

    #[test]
    fn test_vertex_map_errors() -> Result<(), Box<dyn std::error::Error>> {
        let depth = Image::<u16, 1>::from_size_val(TEST_SIZE, 1000)?;
        let mut small = Image::<f32, 3>::from_size_val(
            ImageSize {
                width: TEST_SIZE.height,
                height: TEST_SIZE.width,
            },
            0.0,
        )?;
        // same pixel count, different shape
        assert!(matches!(
            vertex_map_from_depth_mm(&depth, &test_camera(), &mut small),
            Err(RgbdIcpError::SizeMismatch { .. })
        ));
        let distorted = PinholeCamera {
            k1: 0.1,
            ..test_camera()
        };
        let mut verts = Image::<f32, 3>::from_size_val(TEST_SIZE, 0.0)?;
        assert!(matches!(
            vertex_map_from_depth_mm(&depth, &distorted, &mut verts),
            Err(RgbdIcpError::DistortedCamera)
        ));
        for (fx, cx) in [
            (0.0, 1.0),
            (-1.0, 1.0),
            (f64::NAN, 1.0),
            (f64::INFINITY, 1.0),
            (1.0, f64::NAN),
        ] {
            let bad = PinholeCamera {
                fx,
                cx,
                ..test_camera()
            };
            assert!(matches!(
                vertex_map_from_depth_mm(&depth, &bad, &mut verts),
                Err(RgbdIcpError::InvalidIntrinsics { .. })
            ));
        }
        assert!(!is_valid_vertex(&[f32::NAN, 0.0, 1.0]));
        assert!(!is_valid_vertex(&[0.0, 0.0, f32::NAN]));
        assert!(!is_valid_vertex(&[0.0, 0.0, f32::INFINITY]));
        assert!(!is_valid_vertex(&[0.0, 0.0, 0.0]));
        Ok(())
    }

    #[test]
    fn test_tilted_plane_normals_within_one_degree() -> Result<(), Box<dyn std::error::Error>> {
        let camera = test_camera();
        // plane tilted 20 degrees about the x-axis, ~1.5 m in front of the camera
        let (s, c) = (20.0f64).to_radians().sin_cos();
        let scene = Scene {
            planes: vec![Plane {
                normal: Vec3F64::new(0.0, s, c),
                d: 1.5,
            }],
            spheres: vec![],
        };
        // camera-facing ground-truth normal (n_z < 0)
        let n_gt = [0.0, -s, -c];

        // exact (float) vertex map, so the test isolates the cross-product math from u16
        // millimetre quantization
        let verts = render_vertex_map(
            &scene,
            &camera,
            TEST_SIZE,
            &Mat3F64::IDENTITY,
            &Vec3F64::ZERO,
        )?;
        let mut normals = Image::<f32, 3>::from_size_val(TEST_SIZE, 0.0)?;
        normal_map_from_vertex_map(&verts, &mut normals)?;

        let max_angle_rad = 1.0f64.to_radians();
        let mut checked = 0;
        for (idx, n) in normals.as_slice().chunks_exact(3).enumerate() {
            if !is_valid_normal(&[n[0], n[1], n[2]]) {
                continue;
            }
            let dot = (n[0] as f64 * n_gt[0] + n[1] as f64 * n_gt[1] + n[2] as f64 * n_gt[2])
                .clamp(-1.0, 1.0);
            let angle = dot.acos();
            assert!(
                angle < max_angle_rad,
                "normal at pixel {idx} off by {} deg",
                angle.to_degrees()
            );
            checked += 1;
        }
        // all interior pixels see the plane
        assert_eq!(checked, (TEST_SIZE.width - 2) * (TEST_SIZE.height - 2));

        // same pixel count, different shape
        let mut wrong = Image::<f32, 3>::from_size_val(
            ImageSize {
                width: TEST_SIZE.height,
                height: TEST_SIZE.width,
            },
            0.0,
        )?;
        assert!(matches!(
            normal_map_from_vertex_map(&verts, &mut wrong),
            Err(RgbdIcpError::SizeMismatch { .. })
        ));
        Ok(())
    }

    #[test]
    fn test_downsample_holes_and_discontinuities() -> Result<(), Box<dyn std::error::Error>> {
        // 8x8: left half at 1000 mm, right half at 2000 mm, one hole block, one block with a
        // single valid sample
        let (w, h) = (8usize, 8usize);
        let mut depth: Vec<u16> = (0..w * h)
            .map(|i| if i % w < 4 { 1000 } else { 2000 })
            .collect();
        // fully invalid 2x2 block at output (0, 0)
        for i in [0, 1, w, w + 1] {
            depth[i] = 0;
        }
        // single valid sample in block at output (1, 1): keep only (2, 2)
        for i in [2 * w + 3, 3 * w + 2, 3 * w + 3] {
            depth[i] = 0;
        }
        // 3-valid block at output (2, 0) (input x 4..6, y 0..2) straddling the
        // discontinuity: one foreground sample, two background
        depth[w + 4] = 0;
        depth[4] = 1000;
        depth[5] = 2000;
        depth[w + 5] = 2000;

        let src = Image::<u16, 1>::new(
            ImageSize {
                width: w,
                height: h,
            },
            depth,
        )?;
        let half_size = ImageSize {
            width: w / 2,
            height: h / 2,
        };
        let mut dst = Image::<u16, 1>::from_size_val(half_size, 7)?;
        downsample_depth_half(&src, &mut dst)?;
        let (half, hw) = (dst.as_slice(), w / 2);

        // hole block stays invalid: holes never get filled
        assert_eq!(half[0], 0);
        // single valid sample passes through unchanged
        assert_eq!(half[hw + 1], 1000);
        // uniform blocks average exactly
        assert_eq!(half[2 * hw], 1000);
        assert_eq!(half[2 * hw + 3], 2000);
        // mixed 1000/2000 block sides with the median surface, never smears
        assert_eq!(
            half[2], 2000,
            "block with samples [1000, 2000, 2000] must resolve to the median surface"
        );
        // no output pixel anywhere may be an average across the 1 m jump
        for &d in half {
            assert!(
                d == 0 || (d as i32 - 1500).abs() > 400,
                "smeared depth {d} across discontinuity"
            );
        }

        // a 9x9 input halves to 4x4 (floor); a 5x4 output is a shape error
        let odd = Image::<u16, 1>::from_size_val(
            ImageSize {
                width: 9,
                height: 9,
            },
            1000,
        )?;
        downsample_depth_half(&odd, &mut dst)?;
        assert!(dst.as_slice().iter().all(|&d| d == 1000));
        let mut wrong = Image::<u16, 1>::from_size_val(
            ImageSize {
                width: 5,
                height: 4,
            },
            0,
        )?;
        assert!(matches!(
            downsample_depth_half(&src, &mut wrong),
            Err(RgbdIcpError::SizeMismatch { .. })
        ));
        Ok(())
    }

    #[test]
    fn test_pyramid_levels_and_intrinsics() -> Result<(), Box<dyn std::error::Error>> {
        let camera = test_camera();
        let mut depth = render_depth_mm(
            &Scene::corner_and_sphere(),
            &camera,
            TEST_SIZE,
            &Mat3F64::IDENTITY,
            &Vec3F64::ZERO,
        )?;
        // punch a 4x4 hole so coarser levels must keep it invalid
        for y in 60..64 {
            for x in 100..104 {
                depth.as_slice_mut()[y * TEST_SIZE.width + x] = 0;
            }
        }

        let pyr = RgbdPyramid::from_depth_mm(&depth, &camera, 3)?;
        assert_eq!(pyr.levels.len(), 3);

        let mut expected = camera.clone();
        for (i, level) in pyr.levels.iter().enumerate() {
            if i > 0 {
                expected = half_camera(&expected);
            }
            assert_eq!(
                level.camera.intrinsics(),
                expected.intrinsics(),
                "intrinsics at level {i}"
            );
            let n = level.size().width * level.size().height;
            assert_eq!(level.normals.size(), level.size());
            let valid = level
                .vertices
                .as_slice()
                .chunks_exact(3)
                .filter(|v| v[2] > 0.0)
                .count();
            assert!(valid > n / 2, "level {i} should be mostly valid");
        }
        assert_eq!(
            pyr.levels[1].size(),
            ImageSize {
                width: 160,
                height: 90
            }
        );
        assert_eq!(
            pyr.levels[2].size(),
            ImageSize {
                width: 80,
                height: 45
            }
        );
        // cx convention: cx' = (cx + 0.5) / 2 - 0.5
        assert!((pyr.levels[1].camera.cx - ((159.5 + 0.5) / 2.0 - 0.5)).abs() < 1e-12);
        assert!((pyr.levels[1].camera.fx - 100.0).abs() < 1e-12);

        // the hole survives downsampling: the level-1 block covering the hole centre is invalid
        let l1 = &pyr.levels[1];
        assert_eq!(
            l1.vertices.as_slice()[3 * (31 * l1.size().width + 51) + 2],
            0.0
        );

        // a level-1 vertex projects back onto its own pixel with the halved intrinsics
        let idx = 45 * l1.size().width + 80;
        let p = &l1.vertices.as_slice()[3 * idx..3 * idx + 3];
        let uv = l1
            .camera
            .project_to_pixel(&Vec3F64::new(p[0] as f64, p[1] as f64, p[2] as f64), 0.0)
            .ok_or("valid vertex must project")?;
        assert!((uv.x - 80.0).abs() < 1e-3 && (uv.y - 45.0).abs() < 1e-3);

        // errors
        assert!(matches!(
            RgbdPyramid::from_depth_mm(&depth, &camera, 0),
            Err(RgbdIcpError::InvalidNumLevels(0))
        ));
        assert!(matches!(
            RgbdPyramid::from_depth_mm(&depth, &camera, 8),
            Err(RgbdIcpError::ImageTooSmall { level: 6, .. })
        ));
        Ok(())
    }

    /// Metre depth that is `0`, negative or not finite gives an invalid vertex and is ignored
    /// by the halving, as [`is_valid_depth_f32`] decides.
    #[test]
    fn test_metre_depth_invalid_values() -> Result<(), Box<dyn std::error::Error>> {
        let size = ImageSize {
            width: 4,
            height: 2,
        };
        #[rustfmt::skip]
        let depth = Image::<f32, 1>::new(size, vec![
            2.0, 0.0, -1.0, f32::NAN,
            f32::INFINITY, f32::NEG_INFINITY, -0.0, 1.5,
        ])?;
        let camera = PinholeCamera {
            fx: 2.0,
            fy: 2.0,
            cx: 1.5,
            cy: 0.5,
            ..PinholeCamera::IDENTITY
        };
        let mut verts = Image::<f32, 3>::from_size_val(size, 7.0)?;
        vertex_map_from_depth_m(&depth, &camera, &mut verts)?;
        let v = verts.as_slice();
        assert_eq!(&v[..3], &[-1.5, -0.5, 2.0]);
        assert_eq!(&v[21..], &[1.125, 0.375, 1.5]);
        for i in 1..7 {
            assert_eq!(&v[3 * i..3 * i + 3], &[0.0; 3], "pixel {i}");
        }

        // each 2x2 block averages its valid samples only; an all-invalid block becomes 0
        let mut half = Image::<f32, 1>::from_size_val(
            ImageSize {
                width: 2,
                height: 1,
            },
            7.0,
        )?;
        downsample_depth_half_m(&depth, &mut half)?;
        assert_eq!(half.as_slice(), &[2.0, 1.5]);
        let mut all_bad = depth.clone();
        all_bad.as_slice_mut()[0] = f32::NAN;
        all_bad.as_slice_mut()[7] = -2.0;
        downsample_depth_half_m(&all_bad, &mut half)?;
        assert_eq!(half.as_slice(), &[0.0, 0.0]);
        assert!(matches!(
            downsample_depth_half_m(&depth, &mut Image::from_size_val(size, 0.0)?),
            Err(RgbdIcpError::SizeMismatch { .. })
        ));
        Ok(())
    }

    /// A sigma map halves to the sigma of the block mean, and must have the depth's size.
    #[test]
    fn test_sigma_map_pyramid() -> Result<(), Box<dyn std::error::Error>> {
        let size = ImageSize {
            width: 6,
            height: 6,
        };
        let mut depth = Image::<f32, 1>::from_size_val(size, 1.0)?;
        let mut sigma = Image::<f32, 1>::from_size_val(size, 0.004)?;
        // block (0, 0): one invalid depth, three averaged; block (1, 0): one sample without sigma
        depth.as_slice_mut()[0] = f32::NAN;
        sigma.as_slice_mut()[2] = 0.0;
        let pyr = RgbdPyramid::from_depth_m_with_sigma(&depth, &sigma, &test_camera(), 2)?;
        let s1 = pyr.levels[1]
            .depth_sigmas()
            .ok_or("level 1 has a sigma map")?;
        assert_eq!(s1.size(), pyr.levels[1].size());
        let s1 = s1.as_slice();
        assert!((s1[0] - 0.004 / 3f32.sqrt()).abs() < 1e-9);
        assert_eq!(s1[1], 0.0);
        assert!((s1[2] - 0.002).abs() < 1e-9);
        let s0 = pyr.levels[0]
            .depth_sigmas()
            .ok_or("level 0 has a sigma map")?;
        assert_eq!(s0.as_slice(), sigma.as_slice());
        assert!(
            RgbdPyramid::from_depth_m(&depth, &test_camera(), 2)?.levels[0]
                .depth_sigmas()
                .is_none()
        );

        for wrong in [
            ImageSize {
                width: 6,
                height: 5,
            },
            ImageSize {
                width: 5,
                height: 6,
            },
        ] {
            let sigma = Image::<f32, 1>::from_size_val(wrong, 0.004)?;
            assert!(matches!(
                RgbdPyramid::from_depth_m_with_sigma(&depth, &sigma, &test_camera(), 2),
                Err(RgbdIcpError::SizeMismatch { .. })
            ));
        }
        Ok(())
    }

    /// The metre pyramid of a millimetre depth equals the millimetre pyramid: identical `z` at
    /// level 0, and within the millimetre path's rounding of each block mean below it.
    #[test]
    fn test_metre_pyramid_matches_millimetre() -> Result<(), Box<dyn std::error::Error>> {
        let camera = test_camera();
        let depth = render_depth_mm(
            &Scene::corner_and_sphere(),
            &camera,
            TEST_SIZE,
            &Mat3F64::IDENTITY,
            &Vec3F64::ZERO,
        )?;
        let mm = RgbdPyramid::from_depth_mm(&depth, &camera, 3)?;
        let m = RgbdPyramid::from_depth_m(&depth_m_from_mm(&depth)?, &camera, 3)?;
        for (level, (a, b)) in mm.levels.iter().zip(&m.levels).enumerate() {
            assert_eq!(a.camera.intrinsics(), b.camera.intrinsics());
            assert_eq!(a.size(), b.size());
            // half a millimetre of rounding per halving
            let tol_z = 0.5e-3 * level as f32 + 1e-6;
            let (va, vb) = (a.vertices.as_slice(), b.vertices.as_slice());
            for (i, (pa, pb)) in va.chunks_exact(3).zip(vb.chunks_exact(3)).enumerate() {
                assert_eq!(
                    pa[2] > 0.0,
                    pb[2] > 0.0,
                    "validity at level {level}, pixel {i}"
                );
                if level == 0 {
                    assert_eq!(pa[2], pb[2], "z at pixel {i}");
                }
                assert!((pa[2] - pb[2]).abs() <= tol_z, "level {level}, pixel {i}");
                for k in 0..2 {
                    assert!((pa[k] - pb[k]).abs() <= tol_z + 1e-6 * pa[k].abs());
                }
            }
            let (na, nb) = (a.normals.as_slice(), b.normals.as_slice());
            let mut agree = 0;
            for (ma, mb) in na.chunks_exact(3).zip(nb.chunks_exact(3)) {
                assert_eq!(
                    is_valid_normal(&[ma[0], ma[1], ma[2]]),
                    is_valid_normal(&[mb[0], mb[1], mb[2]])
                );
                let dot = ma[0] * mb[0] + ma[1] * mb[1] + ma[2] * mb[2];
                if dot > 0.999 {
                    agree += 1;
                }
            }
            let valid = na.chunks_exact(3).filter(|n| n[2] != 0.0).count();
            assert!(
                agree * 100 >= valid * 99,
                "level {level}: {agree} of {valid}"
            );
        }
        Ok(())
    }

    /// An in-place rebuild equals a fresh build bit for bit, whatever the pyramid held before:
    /// another size, more levels, a sigma map, or other depth at the same size.
    #[test]
    fn rebuild_matches_from_depth_mm() -> Result<(), Box<dyn std::error::Error>> {
        let camera = test_camera();
        let render = |t: Vec3F64| {
            render_depth_mm(
                &Scene::corner_and_sphere(),
                &camera,
                TEST_SIZE,
                &Mat3F64::IDENTITY,
                &t,
            )
        };
        let (first, second) = (
            render(Vec3F64::ZERO)?,
            render(Vec3F64::new(0.03, 0.0, 0.01))?,
        );
        let small = ImageSize {
            width: 40,
            height: 30,
        };
        let stale = Image::<f32, 1>::from_size_val(small, 1.0)?;
        let sigma = Image::<f32, 1>::from_size_val(small, 0.004)?;
        let mut pyr = RgbdPyramid::from_depth_m_with_sigma(&stale, &sigma, &camera, 3)?;
        let mut scratch = PyramidScratch::default();
        let bits = |img: &[f32]| img.iter().map(|x| x.to_bits()).collect::<Vec<_>>();
        for (depth, num_levels) in [(&first, 3), (&second, 3), (&first, 2)] {
            pyr.rebuild_from_depth_mm(depth, &camera, num_levels, &mut scratch)?;
            let fresh = RgbdPyramid::from_depth_mm(depth, &camera, num_levels)?;
            assert_eq!(pyr.levels.len(), num_levels);
            for (a, b) in fresh.levels.iter().zip(&pyr.levels) {
                assert_eq!(a.camera.intrinsics(), b.camera.intrinsics());
                assert_eq!(bits(a.vertices.as_slice()), bits(b.vertices.as_slice()));
                assert_eq!(bits(a.normals.as_slice()), bits(b.normals.as_slice()));
                assert!(b.depth_sigmas.is_none());
            }
        }
        // an invalid input is rejected before anything is written
        let before = bits(pyr.levels[0].vertices.as_slice());
        assert!(matches!(
            pyr.rebuild_from_depth_mm(&first, &camera, 9, &mut scratch),
            Err(RgbdIcpError::ImageTooSmall { .. })
        ));
        assert_eq!(pyr.levels.len(), 2);
        assert_eq!(bits(pyr.levels[0].vertices.as_slice()), before);
        Ok(())
    }
}
