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
//!   [`kornia_imgproc::depth::is_valid_depth_u16`].
//! * Vertex maps: `Image<f32, 3>` of camera-frame points in metres, `[0, 0, 0]` = invalid
//!   (a valid vertex always has `z > 0`).
//! * Normal maps: `Image<f32, 3>` of unit normals oriented towards the camera (`n . v < 0`),
//!   `[0, 0, 0]` = invalid.

use crate::camera::PinholeCamera;
use kornia_algebra::Vec3F32;
use kornia_image::{Image, ImageError, ImageSize};
use kornia_imgproc::depth::is_valid_depth_u16;

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
    check_camera(camera)?;
    check_size(src.size(), dst.size())?;

    let width = src.width();
    let (fx, fy, cx, cy) = camera.intrinsics();
    let rows = src.as_slice().chunks_exact(width);
    let out_rows = dst.as_slice_mut().chunks_exact_mut(3 * width);
    for (v, (row, out_row)) in rows.zip(out_rows).enumerate() {
        for (u, (&d, out_px)) in row.iter().zip(out_row.chunks_exact_mut(3)).enumerate() {
            if !is_valid_depth_u16(d) {
                out_px.fill(0.0);
                continue;
            }
            let z = d as f64 * 1e-3;
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
}

/// Coarse-to-fine pyramid of vertex and normal maps built from one depth image.
///
/// `levels[0]` is the input resolution; each further level halves the depth with
/// [`downsample_depth_half`] and the intrinsics to match.
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
        if num_levels == 0 {
            return Err(RgbdIcpError::InvalidNumLevels(num_levels));
        }
        check_camera(camera)?;

        let mut levels = Vec::with_capacity(num_levels);
        let mut level_camera = camera.clone();
        // level 0 reads `depth` directly; `halved` holds the depth from level 1 on
        let mut halved: Option<Image<u16, 1>> = None;
        for level in 0..num_levels {
            let cur = halved.as_ref().unwrap_or(depth);
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
                let mut half = Image::from_size_val(size, 0u16)?;
                downsample_depth_half(cur, &mut half)?;
                halved = Some(half);
                level_camera = half_camera(&level_camera);
            }
            let cur = halved.as_ref().unwrap_or(depth);
            let mut vertices = Image::from_size_val(size, 0.0f32)?;
            vertex_map_from_depth_mm(cur, &level_camera, &mut vertices)?;
            let mut normals = Image::from_size_val(size, 0.0f32)?;
            normal_map_from_vertex_map(&vertices, &mut normals)?;
            levels.push(RgbdLevel {
                camera: level_camera.clone(),
                vertices,
                normals,
            });
        }

        Ok(Self { levels })
    }
}

#[cfg(test)]
mod tests {
    use super::super::synth::{
        render_depth_mm, render_vertex_map, test_camera, Plane, Scene, TEST_SIZE,
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
}
