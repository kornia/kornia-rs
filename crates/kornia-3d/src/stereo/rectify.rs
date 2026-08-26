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
use kornia_image::{Image, ImageError, ImageSize};
use kornia_imgproc::interpolation::{remap_u8, InterpolationMode};

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

    /// An image operand's resolution does not match the rectifier's.
    #[error("image {got:?} does not match rectifier resolution {expected:?}")]
    ImageSizeMismatch {
        /// Offending image resolution `(width, height)` — source or destination.
        got: (usize, usize),
        /// Rectifier resolution `(width, height)`.
        expected: (usize, usize),
    },

    /// A host byte buffer's length does not match the rectifier's pixel count. The same
    /// caller mistake as [`ImageSizeMismatch`](Self::ImageSizeMismatch), for the raw-slice
    /// entry points — kept typed so a demote-to-CPU policy can classify it as a caller bug,
    /// not a CUDA failure.
    #[error("raw buffer holds {got} bytes, rectifier expects {expected} (width * height)")]
    RawLengthMismatch {
        /// Provided buffer length in bytes.
        got: usize,
        /// Expected length, the rectifier's `width * height`.
        expected: usize,
    },

    /// Failed to build the rectified image from the remapped buffer.
    #[error(transparent)]
    Image(#[from] ImageError),
}

/// Geometry of the rectified rig, shared verbatim by the CPU and CUDA rectifiers so the
/// two cannot drift.
#[derive(Clone, Copy)]
struct RectifiedGeometry {
    width: usize,
    height: usize,
    /// Common rectified focal length (fx = fy).
    f: f64,
    /// Common rectified principal point.
    cx: f64,
    cy: f64,
    /// Metric stereo baseline (‖translation between cameras‖).
    baseline: f64,
    /// Rotation mapping raw left-camera coordinates into the rectified frame.
    rect_left: Mat3F64,
}

impl RectifiedGeometry {
    fn rectified_camera(&self) -> PinholeCamera {
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
}

/// Precomputed stereo rectification for a fixed camera pair and resolution.
pub struct StereoRectifier {
    geom: RectifiedGeometry,
    /// Per-output-pixel source coordinate in the raw left image (`[u, v]`).
    left_map_x: Image<f32, 1>,
    left_map_y: Image<f32, 1>,
    /// Per-output-pixel source coordinate in the raw right image.
    right_map_x: Image<f32, 1>,
    right_map_y: Image<f32, 1>,
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

        let (left_map_x, left_map_y) = build_map(width, height, f, cx, cy, &rect_l, left)?;
        let (right_map_x, right_map_y) = build_map(width, height, f, cx, cy, &rect_r, right)?;

        Ok(Self {
            geom: RectifiedGeometry {
                width,
                height,
                f,
                cx,
                cy,
                baseline: nt,
                rect_left: rect_l,
            },
            left_map_x,
            left_map_y,
            right_map_x,
            right_map_y,
        })
    }

    /// Rotation mapping raw left-camera coordinates into the rectified frame
    /// (`p_rect = R · p_left_raw`). Lets callers re-express raw-frame
    /// extrinsics (e.g. a camera-IMU `T_BS`) for the rectified virtual camera.
    pub fn left_rectifying_rotation(&self) -> Mat3F64 {
        self.geom.rect_left
    }

    /// Rectified pinhole camera (shared by both views; zero distortion).
    pub fn rectified_camera(&self) -> PinholeCamera {
        self.geom.rectified_camera()
    }

    /// The LEFT view's undistort+rectify map as `(x, y)` planes, one f32 per output pixel,
    /// row-major `width * height` — exactly the operands [`remap_u8`] takes, on either backend.
    /// [`rectify_left`](Self::rectify_left) samples through this same table, so an external
    /// consumer sees identical geometry AND identical border/rounding semantics.
    pub fn left_maps(&self) -> (&Image<f32, 1>, &Image<f32, 1>) {
        (&self.left_map_x, &self.left_map_y)
    }

    /// The RIGHT view's map planes; see [`left_maps`](Self::left_maps).
    pub fn right_maps(&self) -> (&Image<f32, 1>, &Image<f32, 1>) {
        (&self.right_map_x, &self.right_map_y)
    }

    /// Metric baseline between the cameras.
    pub fn baseline(&self) -> f64 {
        self.geom.baseline
    }

    /// `bf = focal * baseline`, the constant in `depth = bf / disparity`.
    pub fn bf(&self) -> f64 {
        self.geom.f * self.geom.baseline
    }

    /// Rectifies a raw left image into `dst` — into-style like every imgproc op, so the
    /// output's size and residency are the caller's stated intent, not an allocation policy.
    ///
    /// # Errors
    /// [`StereoError::ImageSizeMismatch`] if `src` or `dst` resolution differs from the
    /// rectifier's.
    pub fn rectify_left(
        &self,
        src: &Image<u8, 1>,
        dst: &mut Image<u8, 1>,
    ) -> Result<(), StereoError> {
        self.remap(src, dst, &self.left_map_x, &self.left_map_y)
    }

    /// Rectifies a raw right image into `dst`; see [`rectify_left`](Self::rectify_left).
    pub fn rectify_right(
        &self,
        src: &Image<u8, 1>,
        dst: &mut Image<u8, 1>,
    ) -> Result<(), StereoError> {
        self.remap(src, dst, &self.right_map_x, &self.right_map_y)
    }

    /// One sampler for every backend: [`remap_u8`], whose CPU and CUDA paths are byte-exact
    /// by tested contract. Border semantics are therefore remap_u8's — coordinates in the
    /// `[w-1, w)` band clamp-sample the edge texel (the previous private sampler left them
    /// black), and blending is Q10 fixed point.
    fn remap(
        &self,
        src: &Image<u8, 1>,
        dst: &mut Image<u8, 1>,
        map_x: &Image<f32, 1>,
        map_y: &Image<f32, 1>,
    ) -> Result<(), StereoError> {
        for img in [src, dst] {
            if (img.width(), img.height()) != (self.geom.width, self.geom.height) {
                return Err(StereoError::ImageSizeMismatch {
                    got: (img.width(), img.height()),
                    expected: (self.geom.width, self.geom.height),
                });
            }
        }
        remap_u8(src, dst, map_x, map_y, InterpolationMode::Bilinear)?;
        Ok(())
    }
}

#[cfg(feature = "cuda")]
pub use cuda::CudaStereoRectifier;

/// The CUDA-resident half, in its own module so the `std::sync::Arc` / cudarc imports exist
/// only when the feature does.
#[cfg(feature = "cuda")]
mod cuda {
    use std::sync::Arc;

    use super::*;

    impl StereoRectifier {
        /// Uploads both eyes' map planes and warms the kernel, returning a rectifier that serves
        /// DEVICE-resident work. Explicit — no hidden H2D on first frame — and the warm-up runs a
        /// full rectify so nvrtc/compile/launch failures surface HERE, where a caller's CPU
        /// fallback can catch them, not on frame one with the fallback already forfeited.
        ///
        /// The stream is only borrowed for the uploads; it is also retained for the host
        /// convenience methods. No context is created — the application owns that.
        pub fn to_cuda(
            &self,
            stream: &Arc<cudarc::driver::CudaStream>,
        ) -> Result<CudaStereoRectifier, StereoError> {
            let size = ImageSize {
                width: self.geom.width,
                height: self.geom.height,
            };
            let mut dev = CudaStereoRectifier {
                geom: self.geom,
                left_map_x: self.left_map_x.to_cuda(stream)?,
                left_map_y: self.left_map_y.to_cuda(stream)?,
                right_map_x: self.right_map_x.to_cuda(stream)?,
                right_map_y: self.right_map_y.to_cuda(stream)?,
                scratch_in: Image::zeros_cuda(size, stream)?,
                scratch_out: Image::zeros_cuda(size, stream)?,
                stream: stream.clone(),
            };
            // Warm-up on the retained device scratch: compiles the kernel through the cache and
            // proves a launch works, with no host round-trip. The synchronize is also load-bearing
            // for correctness — it fences the four async map uploads above, so the maps are
            // quiescent for every later call.
            remap_u8(
                &dev.scratch_in,
                &mut dev.scratch_out,
                &dev.left_map_x,
                &dev.left_map_y,
                InterpolationMode::Bilinear,
            )?;
            stream
                .synchronize()
                .map_err(|e| ImageError::Cuda(format!("warm-up sync: {e}")))?;
            Ok(dev)
        }
    }

    /// [`StereoRectifier`]'s maps, device-resident, rectifying through the SAME
    /// residency-dispatched [`remap_u8`] as the CPU path — the two produce identical bytes by
    /// kornia-imgproc's tested CPU↔CUDA contract.
    ///
    /// Mid-run CUDA errors return typed errors and leave the stream state suspect; the demote-to-CPU
    /// policy belongs to the caller (safe precisely because the bytes match).
    pub struct CudaStereoRectifier {
        geom: RectifiedGeometry,
        left_map_x: Image<f32, 1>,
        left_map_y: Image<f32, 1>,
        right_map_x: Image<f32, 1>,
        right_map_y: Image<f32, 1>,
        scratch_in: Image<u8, 1>,
        scratch_out: Image<u8, 1>,
        stream: Arc<cudarc::driver::CudaStream>,
    }

    impl CudaStereoRectifier {
        /// Rectify a DEVICE-resident left frame into a device-resident destination. Zero copies;
        /// work is enqueued on the images' stream and the caller synchronizes before reading.
        pub fn rectify_left_device(
            &self,
            src: &Image<u8, 1>,
            dst: &mut Image<u8, 1>,
        ) -> Result<(), StereoError> {
            self.check(src)?;
            self.check(dst)?;
            remap_u8(
                src,
                dst,
                &self.left_map_x,
                &self.left_map_y,
                InterpolationMode::Bilinear,
            )?;
            Ok(())
        }

        /// See [`rectify_left_device`](Self::rectify_left_device).
        pub fn rectify_right_device(
            &self,
            src: &Image<u8, 1>,
            dst: &mut Image<u8, 1>,
        ) -> Result<(), StereoError> {
            self.check(src)?;
            self.check(dst)?;
            remap_u8(
                src,
                dst,
                &self.right_map_x,
                &self.right_map_y,
                InterpolationMode::Bilinear,
            )?;
            Ok(())
        }

        /// Host-bytes convenience for the driver loop: H2D into retained scratch, kernel, blocking
        /// D2H into `out` (resized once). No per-frame allocation.
        pub fn rectify_left_into(
            &mut self,
            raw: &[u8],
            out: &mut Vec<u8>,
        ) -> Result<(), StereoError> {
            self.host_roundtrip(raw, out, true)
        }

        /// See [`rectify_left_into`](Self::rectify_left_into).
        pub fn rectify_right_into(
            &mut self,
            raw: &[u8],
            out: &mut Vec<u8>,
        ) -> Result<(), StereoError> {
            self.host_roundtrip(raw, out, false)
        }

        fn host_roundtrip(
            &mut self,
            raw: &[u8],
            out: &mut Vec<u8>,
            left: bool,
        ) -> Result<(), StereoError> {
            let n = self.geom.width * self.geom.height;
            if raw.len() != n {
                return Err(StereoError::RawLengthMismatch {
                    got: raw.len(),
                    expected: n,
                });
            }
            {
                let slice = self
                    .scratch_in
                    .as_cudaslice_mut()
                    .ok_or_else(|| ImageError::Cuda("scratch lost device residency".into()))?;
                self.stream
                    .memcpy_htod(raw, slice)
                    .map_err(|e| ImageError::Cuda(format!("h2d: {e}")))?;
            }
            let (mx, my) = if left {
                (&self.left_map_x, &self.left_map_y)
            } else {
                (&self.right_map_x, &self.right_map_y)
            };
            remap_u8(
                &self.scratch_in,
                &mut self.scratch_out,
                mx,
                my,
                InterpolationMode::Bilinear,
            )?;
            out.resize(n, 0);
            // Blocking: syncs the stream, so the bytes are final when this returns.
            self.scratch_out.to_host_into(out)?;
            Ok(())
        }

        fn check(&self, img: &Image<u8, 1>) -> Result<(), StereoError> {
            if (img.width(), img.height()) != (self.geom.width, self.geom.height) {
                return Err(StereoError::ImageSizeMismatch {
                    got: (img.width(), img.height()),
                    expected: (self.geom.width, self.geom.height),
                });
            }
            // Fail HERE with a typed error: the _device methods are device-only by contract,
            // and nothing downstream compares the MAPS' device to the image's — on a multi-GPU
            // host a cross-device frame would launch with foreign map pointers (illegal memory
            // access), not a typed error.
            let Some(stream) = img.cuda_stream() else {
                return Err(ImageError::Cuda(
                    "rectify_*_device needs a device-resident image (use to_cuda/zeros_cuda)"
                        .into(),
                )
                .into());
            };
            if stream.context().ordinal() != self.stream.context().ordinal() {
                return Err(ImageError::DeviceMismatch.into());
            }
            Ok(())
        }

        /// Rectified pinhole camera (shared by both views; zero distortion).
        pub fn rectified_camera(&self) -> PinholeCamera {
            self.geom.rectified_camera()
        }

        /// Metric baseline between the cameras.
        pub fn baseline(&self) -> f64 {
            self.geom.baseline
        }

        /// `bf = focal * baseline`, the constant in `depth = bf / disparity`.
        pub fn bf(&self) -> f64 {
            self.geom.f * self.geom.baseline
        }

        /// See [`StereoRectifier::left_rectifying_rotation`].
        pub fn left_rectifying_rotation(&self) -> Mat3F64 {
            self.geom.rect_left
        }
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
) -> Result<(Image<f32, 1>, Image<f32, 1>), ImageError> {
    let rect_t = rect.transpose(); // rectified-normalized -> camera-normalized
    let intrinsic = CameraIntrinsic {
        fx: cam.fx,
        fy: cam.fy,
        cx: cam.cx,
        cy: cam.cy,
    };
    let distortion = cam.distortion;

    let mut map_x = vec![0.0f32; width * height];
    let mut map_y = vec![0.0f32; width * height];
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
            map_x[v * width + u] = du as f32;
            map_y[v * width + u] = dv as f32;
        }
    }
    let size = ImageSize { width, height };
    Ok((Image::new(size, map_x)?, Image::new(size, map_y)?))
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
    ) -> Result<Image<u8, 1>, ImageError> {
        let mut buf = vec![0u8; width * height];
        buf[v * width + u] = 255;
        Image::from_size_slice(ImageSize { width, height }, &buf)
    }

    /// Intensity-weighted centroid `(u, v)` of all non-zero pixels.
    fn centroid(img: &Image<u8, 1>) -> Option<(f64, f64)> {
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
            let (_, cvl) = {
                let mut out = Image::from_size_val(img_l.size(), 0u8)?;
                rect.rectify_left(&img_l, &mut out)?;
                centroid(&out).expect("left dot survives rectify")
            };
            let (_, cvr) = {
                let mut out = Image::from_size_val(img_r.size(), 0u8)?;
                rect.rectify_right(&img_r, &mut out)?;
                centroid(&out).expect("right dot survives rectify")
            };

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
        let mut out = Image::from_size_val(wrong.size(), 0u8)?;
        assert!(matches!(
            rect.rectify_left(&wrong, &mut out),
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
        let (mx, my) = rect.left_maps();
        assert_eq!(mx.as_slice().len(), 752 * 480);
        assert_eq!(my.as_slice().len(), 752 * 480);
        Ok(())
    }

    /// CPU and CUDA rectification must produce IDENTICAL bytes: both are remap_u8, whose
    /// backends are byte-exact by kornia-imgproc's tested contract. Requires a CUDA device.
    #[test]
    #[cfg(feature = "cuda")]
    fn cuda_rectify_matches_cpu_byte_exact() -> Result<(), Box<dyn std::error::Error>> {
        use cudarc::driver::CudaContext;
        // Same MH01-style rig as the baseline test: distorted 752x480 pair.
        let left = calib(367.215, 248.375);
        let right = calib(379.999, 255.238);
        let rect = StereoRectifier::from_calib(
            &left,
            &right,
            Mat3F64::IDENTITY,
            Vec3F64::new(-0.11, 0.0, 0.0),
        )?;
        let (w, h) = (752usize, 480usize);
        // Deterministic pseudo-random frame.
        let mut x = 0x9E3779B97F4A7C15u64;
        let raw: Vec<u8> = (0..w * h)
            .map(|_| {
                x ^= x << 13;
                x ^= x >> 7;
                x ^= x << 17;
                (x & 0xFF) as u8
            })
            .collect();
        let img = Image::from_size_slice(
            ImageSize {
                width: w,
                height: h,
            },
            &raw,
        )?;
        let size = ImageSize {
            width: w,
            height: h,
        };
        let mut cpu_l = Image::from_size_val(size, 0u8)?;
        let mut cpu_r = Image::from_size_val(size, 0u8)?;
        rect.rectify_left(&img, &mut cpu_l)?;
        rect.rectify_right(&img, &mut cpu_r)?;

        let ctx = CudaContext::new(0)?;
        let stream = ctx.default_stream();
        let mut dev = rect.to_cuda(&stream)?;
        let mut gpu_l = Vec::new();
        let mut gpu_r = Vec::new();
        dev.rectify_left_into(&raw, &mut gpu_l)?;
        dev.rectify_right_into(&raw, &mut gpu_r)?;

        assert_eq!(cpu_l.as_slice(), gpu_l.as_slice(), "left bytes diverge");
        assert_eq!(cpu_r.as_slice(), gpu_r.as_slice(), "right bytes diverge");
        Ok(())
    }
}
