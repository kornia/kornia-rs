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
use kornia_algebra::{Mat3F64, Vec3F64};
use kornia_image::{allocator::CpuAllocator, Image, ImageSize};
use kornia_imgproc::calibration::distortion::{distort_point_polynomial, PolynomialDistortion};
use kornia_imgproc::calibration::CameraIntrinsic;

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
    pub fn from_calib(
        left: &CameraCalib,
        right: &CameraCalib,
        r_rel: Mat3F64,
        t_rel: Vec3F64,
    ) -> Self {
        let width = left.width;
        let height = left.height;

        // Bouguet: split the relative rotation in half so both cameras rotate
        // symmetrically into a common plane.
        let mut om = rodrigues_vec(&r_rel);
        om *= -0.5;
        let r_r = rodrigues_mat(&om);
        let r_l = r_r.transpose();
        let t = r_r * t_rel;

        // New x-axis along the (rotated) baseline; pick horizontal/vertical.
        let idx = if t.x.abs() > t.y.abs() { 0 } else { 1 };
        let c = component(&t, idx);
        let nt = t.length();
        let mut uu = Vec3F64::ZERO;
        set_component(&mut uu, idx, if c > 0.0 { 1.0 } else { -1.0 });

        let ww = cross(&t, &uu);
        let nw = ww.length();
        let ww = if nw > 0.0 {
            ww * ((c.abs() / nt).clamp(-1.0, 1.0).acos() / nw)
        } else {
            ww
        };
        let w_r = rodrigues_mat(&ww);

        let rect_l = w_r * r_l; // left -> rectified
        let rect_r = w_r * r_r; // right -> rectified

        // Shared rectified intrinsics. Disparity = uL - uR is invariant to the
        // common principal point, so centering the image is safe.
        let f = (left.fx + left.fy + right.fx + right.fy) / 4.0;
        let cx = (width as f64 - 1.0) * 0.5;
        let cy = (height as f64 - 1.0) * 0.5;

        let left_map = build_map(width, height, f, cx, cy, &rect_l, left);
        let right_map = build_map(width, height, f, cx, cy, &rect_r, right);

        Self {
            width,
            height,
            f,
            cx,
            cy,
            baseline: nt,
            left_map,
            right_map,
        }
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
    pub fn rectify_left(&self, img: &Image<u8, 1, CpuAllocator>) -> Image<u8, 1, CpuAllocator> {
        self.remap(img, &self.left_map)
    }

    /// Rectifies a raw right image.
    pub fn rectify_right(&self, img: &Image<u8, 1, CpuAllocator>) -> Image<u8, 1, CpuAllocator> {
        self.remap(img, &self.right_map)
    }

    fn remap(
        &self,
        img: &Image<u8, 1, CpuAllocator>,
        map: &[[f32; 2]],
    ) -> Image<u8, 1, CpuAllocator> {
        let src = img.as_slice();
        let (sw, sh) = (img.width(), img.height());
        let mut out = vec![0u8; self.width * self.height];
        for (dst, &[fx, fy]) in out.iter_mut().zip(map.iter()) {
            *dst = sample_bilinear(src, sw, sh, fx, fy);
        }
        Image::from_size_slice(
            ImageSize {
                width: self.width,
                height: self.height,
            },
            &out,
            CpuAllocator,
        )
        .expect("rectified image size matches buffer")
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

fn sample_bilinear(src: &[u8], w: usize, h: usize, fx: f32, fy: f32) -> u8 {
    if !fx.is_finite() || !fy.is_finite() || fx < 0.0 || fy < 0.0 {
        return 0;
    }
    let (wf, hf) = (w as f32, h as f32);
    if fx > wf - 1.0 || fy > hf - 1.0 {
        return 0;
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
    (top + (bot - top) * ay).round().clamp(0.0, 255.0) as u8
}

/// Rotation vector -> rotation matrix (Rodrigues).
fn rodrigues_mat(v: &Vec3F64) -> Mat3F64 {
    let theta = v.length();
    if theta < 1e-12 {
        return Mat3F64::IDENTITY;
    }
    let k = *v / theta;
    let kx = skew(&k);
    let kx2 = kx * kx;
    Mat3F64::IDENTITY + kx * theta.sin() + kx2 * (1.0 - theta.cos())
}

/// Rotation matrix -> rotation vector (Rodrigues).
fn rodrigues_vec(m: &Mat3F64) -> Vec3F64 {
    let e = |r: usize, c: usize| mat_elem(m, r, c);
    let trace = e(0, 0) + e(1, 1) + e(2, 2);
    let cos_theta = ((trace - 1.0) * 0.5).clamp(-1.0, 1.0);
    let theta = cos_theta.acos();
    if theta < 1e-12 {
        return Vec3F64::ZERO;
    }
    let axis = Vec3F64::new(e(2, 1) - e(1, 2), e(0, 2) - e(2, 0), e(1, 0) - e(0, 1));
    let s = 2.0 * theta.sin();
    axis * (theta / s)
}

/// Skew-symmetric cross-product matrix of a vector.
fn skew(k: &Vec3F64) -> Mat3F64 {
    Mat3F64::from_cols(
        Vec3F64::new(0.0, k.z, -k.y),
        Vec3F64::new(-k.z, 0.0, k.x),
        Vec3F64::new(k.y, -k.x, 0.0),
    )
}

/// Element at (row `r`, col `c`) of a column-major 3x3 matrix.
fn mat_elem(m: &Mat3F64, r: usize, c: usize) -> f64 {
    let col = match c {
        0 => m.x_axis(),
        1 => m.y_axis(),
        _ => m.z_axis(),
    };
    match r {
        0 => col.x,
        1 => col.y,
        _ => col.z,
    }
}

fn cross(a: &Vec3F64, b: &Vec3F64) -> Vec3F64 {
    Vec3F64::new(
        a.y * b.z - a.z * b.y,
        a.z * b.x - a.x * b.z,
        a.x * b.y - a.y * b.x,
    )
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
    fn rodrigues_round_trip() {
        let v = Vec3F64::new(0.1, -0.2, 0.05);
        let m = rodrigues_mat(&v);
        let back = rodrigues_vec(&m);
        assert!((back - v).length() < 1e-9, "got {back:?}");
    }

    #[test]
    fn rectified_baseline_matches_mh01() {
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
        let rect = StereoRectifier::from_calib(&left, &right, r_rel, t_rel);

        // EuRoC VI-sensor stereo baseline is ~0.11 m.
        assert!(
            (rect.baseline() - 0.11).abs() < 0.01,
            "baseline {} not ~0.11 m",
            rect.baseline()
        );
        assert!(rect.bf() > 0.0);
        assert_eq!(rect.left_map.len(), 752 * 480);
    }
}
