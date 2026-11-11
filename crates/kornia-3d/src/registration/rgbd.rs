//! Depth-image (RGB-D) geometry for projective registration: vertex maps.
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

use crate::camera::PinholeCamera;
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

#[cfg(test)]
mod tests {
    use super::super::synth::{render_depth_mm, test_camera, Scene, TEST_SIZE};
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
}
