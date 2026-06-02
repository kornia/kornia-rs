//! Bouguet stereo rectification for a non-row-aligned camera pair.
//!
//! Many stereo rigs ship raw left/right images with independent intrinsics,
//! distortion, and a relative pose — not row-aligned. Stereo matching assumes a
//! rectified, row-aligned pair, so we rectify: compute Bouguet rectification
//! rotations, build a per-pixel undistort+rectify remap for each view, and
//! resample the raw images.
//!
//! Mirrors OpenCV's `stereoRectify` + `initUndistortRectifyMap` + `remap`.
//!
//! This core is dataset-agnostic: it consumes a generic [`CameraCalib`] and the
//! relative pose between the two cameras. Dataset adapters (e.g. EuRoC `T_BS`
//! extrinsics) live with their callers and feed [`StereoRectifier::from_calib`].

use crate::camera::PinholeCamera;
use kornia_algebra::{Mat3F64, Vec3F64, SO3F64};
use kornia_image::{allocator::CpuAllocator, Image, ImageError, ImageSize};
use kornia_imgproc::calibration::distortion::{distort_point_polynomial, PolynomialDistortion};
use kornia_imgproc::calibration::CameraIntrinsic;

/// Errors produced while building or applying a [`StereoRectifier`].
#[derive(Debug, thiserror::Error)]
pub enum StereoError {
    /// The left and right calibrations describe different resolutions.
    #[error(
        "left and right calibration must share the same resolution, got {left:?} and {right:?}"
    )]
    ResolutionMismatch {
        /// Left calibration resolution `(width, height)`.
        left: (usize, usize),
        /// Right calibration resolution `(width, height)`.
        right: (usize, usize),
    },

    /// The stereo baseline is (near) zero, so no rectifying rotation exists.
    #[error(
        "degenerate stereo baseline: |t_rel| = {0} m; the cameras must be physically separated"
    )]
    DegenerateBaseline(f64),

    /// An input image's resolution does not match the rectifier's.
    #[error("input image {got:?} does not match rectifier resolution {expected:?}")]
    ImageSizeMismatch {
        /// Provided image resolution `(width, height)`.
        got: (usize, usize),
        /// Rectifier resolution `(width, height)`.
        expected: (usize, usize),
    },

    /// Failed to build the rectified image from the remapped buffer.
    #[error(transparent)]
    Image(#[from] ImageError),
}

/// Precomputed stereo rectification for a fixed camera pair and resolution.
pub struct StereoRectifier {
    width: usize,
    height: usize,
    /// Common rectified focal length (fx = fy).
    f: f64,
    /// Common rectified principal point.
    cx: f64,
    cy: f64,
    /// Metric stereo baseline (‖translation between cameras‖).
    baseline: f64,
    /// Per-output-pixel source coordinate in the raw left image (`[u, v]`).
    left_map: Vec<[f32; 2]>,
    /// Per-output-pixel source coordinate in the raw right image.
    right_map: Vec<[f32; 2]>,
}

/// Per-camera calibration for rectification: intrinsics + Brown-Conrady
/// (rational) distortion at a fixed resolution. Decouples the rectifier from
/// any one dataset's calibration container.
pub struct CameraCalib {
    /// Image width in pixels.
    pub width: usize,
    /// Image height in pixels.
    pub height: usize,
    /// Focal length x (pixels).
    pub fx: f64,
    /// Focal length y (pixels).
    pub fy: f64,
    /// Principal point x (pixels).
    pub cx: f64,
    /// Principal point y (pixels).
    pub cy: f64,
    /// Lens distortion (radial k1-k6 + tangential p1,p2).
    pub distortion: PolynomialDistortion,
}

impl StereoRectifier {
    /// Builds the rectifier from generic left/right calibration and the
    /// relative pose left → right (`X_right = r_rel * X_left + t_rel`, with
    /// `t_rel` in metres).
    ///
    /// # Errors
    /// - [`StereoError::ResolutionMismatch`] if `left` and `right` describe
    ///   different resolutions.
    /// - [`StereoError::DegenerateBaseline`] if `t_rel` is (near) zero — a
    ///   degenerate baseline has no well-defined rectifying rotation and would
    ///   otherwise produce `NaN` remap tables.
    pub fn from_calib(
        left: &CameraCalib,
        right: &CameraCalib,
        r_rel: Mat3F64,
        t_rel: Vec3F64,
    ) -> Result<Self, StereoError> {
        if (left.width, left.height) != (right.width, right.height) {
            return Err(StereoError::ResolutionMismatch {
                left: (left.width, left.height),
                right: (right.width, right.height),
            });
        }

        let width = left.width;
        let height = left.height;

        // Bouguet: split the relative rotation in half so both cameras rotate
        // symmetrically into a common plane.
        let mut om = SO3F64::from_matrix(&r_rel).log();
        om *= -0.5;
        let r_r = SO3F64::exp(om).matrix();
        let r_l = r_r.transpose();
        let t = r_r * t_rel;

        // New x-axis along the (rotated) baseline; pick horizontal/vertical.
        let idx = if t.x.abs() > t.y.abs() { 0 } else { 1 };
        let c = component(&t, idx);
        let nt = t.length();
        if nt <= 1e-9 {
            return Err(StereoError::DegenerateBaseline(nt));
        }
        let mut uu = Vec3F64::ZERO;
        set_component(&mut uu, idx, if c > 0.0 { 1.0 } else { -1.0 });

        let ww = t.cross(uu);
        let nw = ww.length();
        let ww = if nw > 0.0 {
            ww * ((c.abs() / nt).clamp(-1.0, 1.0).acos() / nw)
        } else {
            ww
        };
        let w_r = SO3F64::exp(ww).matrix();

        let rect_l = w_r * r_l; // left -> rectified
        let rect_r = w_r * r_r; // right -> rectified

        // Shared rectified intrinsics. Disparity = uL - uR is invariant to the
        // common principal point, so centering the image is safe.
        let f = (left.fx + left.fy + right.fx + right.fy) / 4.0;
        let cx = (width as f64 - 1.0) * 0.5;
        let cy = (height as f64 - 1.0) * 0.5;

        let left_map = build_map(width, height, f, cx, cy, &rect_l, left);
        let right_map = build_map(width, height, f, cx, cy, &rect_r, right);

        Ok(Self {
            width,
            height,
            f,
            cx,
            cy,
            baseline: nt,
            left_map,
            right_map,
        })
    }

    /// Rectified pinhole camera (shared by both views; zero distortion).
    pub fn rectified_camera(&self) -> PinholeCamera {
        PinholeCamera {
            fx: self.f,
            fy: self.f,
            cx: self.cx,
            cy: self.cy,
            k1: 0.0,
            k2: 0.0,
            p1: 0.0,
            p2: 0.0,
        }
    }

    /// Metric baseline between the cameras.
    pub fn baseline(&self) -> f64 {
        self.baseline
    }

    /// `bf = focal * baseline`, the constant in `depth = bf / disparity`.
    pub fn bf(&self) -> f64 {
        self.f * self.baseline
    }

    /// Rectifies a raw left image.
    ///
    /// # Errors
    /// [`StereoError::ImageSizeMismatch`] if `img`'s resolution differs from
    /// the rectifier's.
    pub fn rectify_left(
        &self,
        img: &Image<u8, 1, CpuAllocator>,
    ) -> Result<Image<u8, 1, CpuAllocator>, StereoError> {
        self.remap(img, &self.left_map)
    }

    /// Rectifies a raw right image.
    ///
    /// # Errors
    /// [`StereoError::ImageSizeMismatch`] if `img`'s resolution differs from
    /// the rectifier's.
    pub fn rectify_right(
        &self,
        img: &Image<u8, 1, CpuAllocator>,
    ) -> Result<Image<u8, 1, CpuAllocator>, StereoError> {
        self.remap(img, &self.right_map)
    }

    fn remap(
        &self,
        img: &Image<u8, 1, CpuAllocator>,
        map: &[[f32; 2]],
    ) -> Result<Image<u8, 1, CpuAllocator>, StereoError> {
        if (img.width(), img.height()) != (self.width, self.height) {
            return Err(StereoError::ImageSizeMismatch {
                got: (img.width(), img.height()),
                expected: (self.width, self.height),
            });
        }
        let src = img.as_slice();
        let (sw, sh) = (img.width(), img.height());
        // `out` starts black; only pixels whose source lands inside the raw
        // image get written, so out-of-frame borders stay black.
        let mut out = vec![0u8; self.width * self.height];
        for (dst, &[fx, fy]) in out.iter_mut().zip(map.iter()) {
            if let Some(v) = sample_bilinear(src, sw, sh, fx, fy) {
                *dst = v;
            }
        }
        Ok(Image::from_size_slice(
            ImageSize {
                width: self.width,
                height: self.height,
            },
            &out,
            CpuAllocator,
        )?)
    }
}

/// Builds the undistort+rectify map: for each rectified output pixel, the
/// source coordinate in the raw (distorted) image of that camera.
fn build_map(
    width: usize,
    height: usize,
    f: f64,
    cx: f64,
    cy: f64,
    rect: &Mat3F64,
    cam: &CameraCalib,
) -> Vec<[f32; 2]> {
    let rect_t = rect.transpose(); // rectified-normalized -> camera-normalized
    let intrinsic = CameraIntrinsic {
        fx: cam.fx,
        fy: cam.fy,
        cx: cam.cx,
        cy: cam.cy,
    };
    let distortion = cam.distortion;

    let mut map = vec![[0.0f32; 2]; width * height];
    for v in 0..height {
        for u in 0..width {
            // Inverse rectified projection -> normalized rectified coords.
            let xr = (u as f64 - cx) / f;
            let yr = (v as f64 - cy) / f;
            // Back-rotate into the camera frame, normalize.
            let p = rect_t * Vec3F64::new(xr, yr, 1.0);
            let xn = p.x / p.z;
            let yn = p.y / p.z;
            // Project to an ideal pixel under the camera's K, then distort.
            let px = cam.fx * xn + cam.cx;
            let py = cam.fy * yn + cam.cy;
            let (du, dv) = distort_point_polynomial(px, py, &intrinsic, &distortion);
            map[v * width + u] = [du as f32, dv as f32];
        }
    }
    map
}

/// Samples the source image at `(fx, fy)` with bilinear interpolation.
/// Returns `None` when the coordinate is non-finite or outside the image, so
/// the caller decides how to fill those (out-of-frame) rectified pixels.
fn sample_bilinear(src: &[u8], w: usize, h: usize, fx: f32, fy: f32) -> Option<u8> {
    if !fx.is_finite() || !fy.is_finite() || fx < 0.0 || fy < 0.0 {
        return None;
    }
    let (wf, hf) = (w as f32, h as f32);
    if fx > wf - 1.0 || fy > hf - 1.0 {
        return None;
    }
    let x0 = fx.floor() as usize;
    let y0 = fy.floor() as usize;
    let x1 = (x0 + 1).min(w - 1);
    let y1 = (y0 + 1).min(h - 1);
    let ax = fx - x0 as f32;
    let ay = fy - y0 as f32;
    let p00 = src[y0 * w + x0] as f32;
    let p01 = src[y0 * w + x1] as f32;
    let p10 = src[y1 * w + x0] as f32;
    let p11 = src[y1 * w + x1] as f32;
    let top = p00 + (p01 - p00) * ax;
    let bot = p10 + (p11 - p10) * ax;
    Some((top + (bot - top) * ay).round().clamp(0.0, 255.0) as u8)
}

fn component(v: &Vec3F64, idx: usize) -> f64 {
    if idx == 0 {
        v.x
    } else {
        v.y
    }
}

fn set_component(v: &mut Vec3F64, idx: usize, val: f64) {
    if idx == 0 {
        v.x = val;
    } else {
        v.y = val;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn calib(cx: f64, cy: f64) -> CameraCalib {
        CameraCalib {
            width: 752,
            height: 480,
            fx: 458.0,
            fy: 457.0,
            cx,
            cy,
            distortion: PolynomialDistortion {
                k1: -0.28,
                k2: 0.07,
                k3: 0.0,
                k4: 0.0,
                k5: 0.0,
                k6: 0.0,
                p1: 0.0,
                p2: 0.0,
            },
        }
    }

    /// Pinhole calibration with no lens distortion.
    fn pinhole(width: usize, height: usize, f: f64, cx: f64, cy: f64) -> CameraCalib {
        CameraCalib {
            width,
            height,
            fx: f,
            fy: f,
            cx,
            cy,
            distortion: PolynomialDistortion {
                k1: 0.0,
                k2: 0.0,
                k3: 0.0,
                k4: 0.0,
                k5: 0.0,
                k6: 0.0,
                p1: 0.0,
                p2: 0.0,
            },
        }
    }

    /// Projects a 3D camera-frame point to a pixel (distortion-free pinhole).
    fn project(cam: &CameraCalib, p: Vec3F64) -> Option<(f64, f64)> {
        if p.z <= 0.0 {
            return None;
        }
        Some((cam.fx * p.x / p.z + cam.cx, cam.fy * p.y / p.z + cam.cy))
    }

    /// A black image with a single white pixel at `(u, v)`.
    fn dot_image(
        width: usize,
        height: usize,
        u: usize,
        v: usize,
    ) -> Result<Image<u8, 1, CpuAllocator>, ImageError> {
        let mut buf = vec![0u8; width * height];
        buf[v * width + u] = 255;
        Image::from_size_slice(ImageSize { width, height }, &buf, CpuAllocator)
    }

    /// Intensity-weighted centroid `(u, v)` of all non-zero pixels.
    fn centroid(img: &Image<u8, 1, CpuAllocator>) -> Option<(f64, f64)> {
        let w = img.width();
        let (mut su, mut sv, mut sw) = (0.0, 0.0, 0.0);
        for (i, &p) in img.as_slice().iter().enumerate() {
            if p == 0 {
                continue;
            }
            let val = p as f64;
            su += val * (i % w) as f64;
            sv += val * (i / w) as f64;
            sw += val;
        }
        (sw > 0.0).then(|| (su / sw, sv / sw))
    }

    /// The defining property of rectification: the same 3D point projects to the
    /// same *row* in both rectified images (the epipolar lines are horizontal and
    /// aligned). This exercises the full rectification pipeline — split rotation,
    /// `w_r` construction, and `build_map` — which `baseline()` alone does not.
    #[test]
    fn rectified_views_are_row_aligned() -> Result<(), Box<dyn std::error::Error>> {
        let (w, h) = (640, 480);
        let left = pinhole(w, h, 400.0, 320.0, 240.0);
        let right = pinhole(w, h, 400.0, 320.0, 240.0);

        // A non-trivial relative pose: ~2° pitch + ~1° yaw, 0.1 m baseline. The
        // pitch makes the *raw* rows disagree by ~14 px, so a broken rectifier
        // would fail this by a wide margin.
        let r_rel = SO3F64::exp(Vec3F64::new(0.035, 0.018, 0.0)).matrix();
        let t_rel = Vec3F64::new(-0.10, 0.0, 0.0);
        let rect = StereoRectifier::from_calib(&left, &right, r_rel, t_rel)?;

        // Points in the left camera frame, all comfortably inside the frustum.
        let points = [
            Vec3F64::new(0.0, 0.0, 3.0),
            Vec3F64::new(0.10, -0.05, 4.0),
            Vec3F64::new(-0.08, 0.06, 2.5),
        ];

        let mut checked = 0;
        for x_left in points {
            let x_right = r_rel * x_left + t_rel;
            let (Some((ul, vl)), Some((ur, vr))) =
                (project(&left, x_left), project(&right, x_right))
            else {
                continue;
            };
            // Stamp each raw view and rectify.
            let img_l = dot_image(w, h, ul.round() as usize, vl.round() as usize)?;
            let img_r = dot_image(w, h, ur.round() as usize, vr.round() as usize)?;
            let (_, cvl) =
                centroid(&rect.rectify_left(&img_l)?).expect("left dot survives rectify");
            let (_, cvr) =
                centroid(&rect.rectify_right(&img_r)?).expect("right dot survives rectify");

            assert!(
                (cvl - cvr).abs() < 2.0,
                "rectified rows not aligned: left v={cvl}, right v={cvr}"
            );
            checked += 1;
        }
        assert!(checked >= 2, "too few points stayed in frame ({checked})");
        Ok(())
    }

    #[test]
    fn mismatched_resolution_errors() {
        let left = pinhole(640, 480, 400.0, 320.0, 240.0);
        let right = pinhole(752, 480, 400.0, 320.0, 240.0);
        let result = StereoRectifier::from_calib(
            &left,
            &right,
            Mat3F64::IDENTITY,
            Vec3F64::new(-0.1, 0.0, 0.0),
        );
        assert!(matches!(
            result,
            Err(StereoError::ResolutionMismatch { .. })
        ));
    }

    #[test]
    fn zero_baseline_errors() {
        let left = pinhole(640, 480, 400.0, 320.0, 240.0);
        let right = pinhole(640, 480, 400.0, 320.0, 240.0);
        let result = StereoRectifier::from_calib(&left, &right, Mat3F64::IDENTITY, Vec3F64::ZERO);
        assert!(matches!(result, Err(StereoError::DegenerateBaseline(_))));
    }

    #[test]
    fn rectify_size_mismatch_errors() -> Result<(), Box<dyn std::error::Error>> {
        let left = pinhole(640, 480, 400.0, 320.0, 240.0);
        let right = pinhole(640, 480, 400.0, 320.0, 240.0);
        let rect = StereoRectifier::from_calib(
            &left,
            &right,
            Mat3F64::IDENTITY,
            Vec3F64::new(-0.1, 0.0, 0.0),
        )?;
        let wrong = dot_image(320, 240, 10, 10)?;
        assert!(matches!(
            rect.rectify_left(&wrong),
            Err(StereoError::ImageSizeMismatch { .. })
        ));
        Ok(())
    }

    /// Splits a row-major 4x4 `T_BS` into rotation (3x3) and translation (3).
    fn decompose_t_bs(m: &[f64; 16]) -> (Mat3F64, Vec3F64) {
        let r = Mat3F64::from_cols(
            Vec3F64::new(m[0], m[4], m[8]),
            Vec3F64::new(m[1], m[5], m[9]),
            Vec3F64::new(m[2], m[6], m[10]),
        );
        let t = Vec3F64::new(m[3], m[7], m[11]);
        (r, t)
    }

    /// Relative pose left -> right (`X_right = r * X_left + t`) from two
    /// body-frame extrinsics.
    fn relative_pose(t_bs_l: &[f64; 16], t_bs_r: &[f64; 16]) -> (Mat3F64, Vec3F64) {
        let (r_l, t_l) = decompose_t_bs(t_bs_l);
        let (r_r, t_r) = decompose_t_bs(t_bs_r);
        let r_rt = r_r.transpose();
        (r_rt * r_l, r_rt * (t_l - t_r))
    }

    #[test]
    fn rectified_baseline_matches_mh01() -> Result<(), Box<dyn std::error::Error>> {
        // Real MH_01_easy cam0/cam1 T_BS (row-major) and principal points.
        let t_bs0 = [
            0.0148655429818,
            -0.999880929698,
            0.00414029679422,
            -0.0216401454975,
            0.999557249008,
            0.0149672133247,
            0.025715529948,
            -0.064676986768,
            -0.0257744366974,
            0.00375618835797,
            0.999660727178,
            0.00981073058949,
            0.0,
            0.0,
            0.0,
            1.0,
        ];
        let t_bs1 = [
            0.0125552670891,
            -0.999755099723,
            0.0182237714554,
            -0.0198435579556,
            0.999598781151,
            0.0130119051815,
            0.0251588363115,
            0.0453689425024,
            -0.0253898008918,
            0.0179005838253,
            0.999517347078,
            0.00786212447038,
            0.0,
            0.0,
            0.0,
            1.0,
        ];
        let left = calib(367.215, 248.375);
        let right = calib(379.999, 255.238);
        let (r_rel, t_rel) = relative_pose(&t_bs0, &t_bs1);
        let rect = StereoRectifier::from_calib(&left, &right, r_rel, t_rel)?;

        // EuRoC VI-sensor stereo baseline is ~0.11 m.
        assert!(
            (rect.baseline() - 0.11).abs() < 0.01,
            "baseline {} not ~0.11 m",
            rect.baseline()
        );
        assert!(rect.bf() > 0.0);
        assert_eq!(rect.left_map.len(), 752 * 480);
        Ok(())
    }
}
