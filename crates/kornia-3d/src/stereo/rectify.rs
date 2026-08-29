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
use kornia_imgproc::interpolation::grid::meshgrid_from_fn;

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

    /// An underlying image or CUDA operation failed (allocation, upload, residency, or
    /// resampling).
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
            ..PinholeCamera::IDENTITY
        }
    }

    fn bf(&self) -> f64 {
        self.f * self.baseline
    }

    /// The one home of the size gate both backends apply to every image operand.
    fn check_size(&self, img: &Image<u8, 1>) -> Result<(), StereoError> {
        if (img.width(), img.height()) != (self.width, self.height) {
            return Err(StereoError::ImageSizeMismatch {
                got: (img.width(), img.height()),
                expected: (self.width, self.height),
            });
        }
        Ok(())
    }
}

/// One eye's undistort+rectify map as x/y planes — the pairing invariant (same size,
/// built together, consumed together) lives here rather than in field-naming convention.
struct MapPair {
    /// Per-output-pixel source COLUMN in the raw image.
    x: Image<f32, 1>,
    /// Per-output-pixel source ROW in the raw image.
    y: Image<f32, 1>,
}

/// Precomputed stereo rectification for a fixed camera pair and resolution.
pub struct StereoRectifier {
    geom: RectifiedGeometry,
    /// The LEFT view's map.
    left_map: MapPair,
    /// The RIGHT view's map.
    right_map: MapPair,
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

        let left_map = build_map(width, height, f, cx, cy, &rect_l, left)?;
        let right_map = build_map(width, height, f, cx, cy, &rect_r, right)?;

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
            left_map,
            right_map,
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
        (&self.left_map.x, &self.left_map.y)
    }

    /// The RIGHT view's map planes; see [`left_maps`](Self::left_maps).
    pub fn right_maps(&self) -> (&Image<f32, 1>, &Image<f32, 1>) {
        (&self.right_map.x, &self.right_map.y)
    }

    /// Metric baseline between the cameras.
    pub fn baseline(&self) -> f64 {
        self.geom.baseline
    }

    /// `bf = focal * baseline`, the constant in `depth = bf / disparity`.
    pub fn bf(&self) -> f64 {
        self.geom.bf()
    }

    /// Rectifies a raw left image into `dst` — into-style per imgproc's convention, so the
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
        self.remap(src, dst, &self.left_map)
    }

    /// Rectifies a raw right image into `dst`; see [`rectify_left`](Self::rectify_left).
    ///
    /// # Errors
    /// Same as [`rectify_left`](Self::rectify_left).
    pub fn rectify_right(
        &self,
        src: &Image<u8, 1>,
        dst: &mut Image<u8, 1>,
    ) -> Result<(), StereoError> {
        self.remap(src, dst, &self.right_map)
    }

    /// One sampler for every backend: [`remap_u8`], whose CPU and CUDA paths are byte-exact
    /// by tested contract. Border semantics are therefore remap_u8's — coordinates in the
    /// OPEN `(w-1, w)` band clamp-sample the edge texel (the previous private sampler left
    /// them black; exactly `w-1` sampled the edge under both), and blending is Q10 fixed
    /// point.
    fn remap(
        &self,
        src: &Image<u8, 1>,
        dst: &mut Image<u8, 1>,
        map: &MapPair,
    ) -> Result<(), StereoError> {
        self.geom.check_size(src)?;
        self.geom.check_size(dst)?;
        remap_u8(src, dst, &map.x, &map.y, InterpolationMode::Bilinear)?;
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

    impl MapPair {
        fn to_cuda(&self, stream: &Arc<cudarc::driver::CudaStream>) -> Result<Self, ImageError> {
            Ok(Self {
                x: self.x.to_cuda(stream)?,
                y: self.y.to_cuda(stream)?,
            })
        }
    }

    impl StereoRectifier {
        /// Uploads both eyes' map planes and warms the kernel, returning a rectifier that serves
        /// DEVICE-resident work. Explicit — no hidden H2D on first frame — and the warm-up runs a
        /// full rectify so nvrtc compile failures surface HERE, where a caller's CPU fallback
        /// can catch them, not on frame one with the fallback already forfeited.
        ///
        /// Map uploads and the warm-up launch are enqueued on `stream`, and the stream is
        /// SYNCHRONIZED before returning: unlike per-frame work, the maps must be globally
        /// visible — a consumer holding a DIFFERENT `CudaContext` instance gets no automatic
        /// ordering (cudarc's auto-fence is per context instance), and an unfenced launch
        /// there would race the uploads and read half-written tables, silently. One host
        /// sync at build time removes the trap; per-frame calls stay unsynchronized.
        /// No context is created — the application owns that.
        ///
        /// # Errors
        /// Upload/allocation failures and the warm-up's nvrtc compile surface here as
        /// [`StereoError::Image`]`(`[`ImageError::Cuda`]`)` — deliberately, so a caller's
        /// CPU fallback can catch them before the first real frame.
        pub fn to_cuda(
            &self,
            stream: &Arc<cudarc::driver::CudaStream>,
        ) -> Result<CudaStereoRectifier, StereoError> {
            let size = ImageSize {
                width: self.geom.width,
                height: self.geom.height,
            };
            let dev = CudaStereoRectifier {
                geom: self.geom,
                left_map: self.left_map.to_cuda(stream)?,
                right_map: self.right_map.to_cuda(stream)?,
                stream: stream.clone(),
            };
            // Warm-up on throwaway device buffers, through the REAL entry point so it compiles
            // the exact kernel later calls launch (a host-side nvrtc step — failures surface
            // synchronously) and exercises the same residency checks.
            let warm_src: Image<u8, 1> = Image::zeros_cuda(size, stream)?;
            // SAFETY: the warm-up rectify writes every output pixel before anything reads
            // `warm_dst`, which is the exact contract `uninit_cuda` asks for — this skips a
            // full-frame memset. `warm_src` stays zeroed: the kernel READS it.
            let mut warm_dst = unsafe { Image::uninit_cuda(size, stream)? };
            dev.rectify_left_device(&warm_src, &mut warm_dst)?;
            stream
                .synchronize()
                .map_err(|e| ImageError::Cuda(e.to_string()))?;
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
        left_map: MapPair,
        right_map: MapPair,
        stream: Arc<cudarc::driver::CudaStream>,
    }

    impl CudaStereoRectifier {
        /// Rectify a DEVICE-resident left frame into a device-resident destination. Zero copies;
        /// work is enqueued on the images' stream and the caller synchronizes before reading.
        ///
        /// # Errors
        /// [`StereoError::ImageSizeMismatch`] on a wrong-resolution operand;
        /// [`ImageError::HostResident`] if either image is not device-resident (a routing
        /// bug, distinct from [`ImageError::Cuda`]'s genuine driver failures);
        /// [`ImageError::DeviceMismatch`] on a cross-device operand;
        /// [`ImageError::Cuda`] for launch failures.
        pub fn rectify_left_device(
            &self,
            src: &Image<u8, 1>,
            dst: &mut Image<u8, 1>,
        ) -> Result<(), StereoError> {
            self.remap_device(src, dst, &self.left_map)
        }

        /// See [`rectify_left_device`](Self::rectify_left_device).
        ///
        /// # Errors
        /// Same as [`rectify_left_device`](Self::rectify_left_device).
        pub fn rectify_right_device(
            &self,
            src: &Image<u8, 1>,
            dst: &mut Image<u8, 1>,
        ) -> Result<(), StereoError> {
            self.remap_device(src, dst, &self.right_map)
        }

        /// The device twin of [`StereoRectifier::remap`]: checks, then one `remap_u8` call.
        fn remap_device(
            &self,
            src: &Image<u8, 1>,
            dst: &mut Image<u8, 1>,
            map: &MapPair,
        ) -> Result<(), StereoError> {
            self.check(src)?;
            self.check(dst)?;
            remap_u8(src, dst, &map.x, &map.y, InterpolationMode::Bilinear)?;
            Ok(())
        }

        fn check(&self, img: &Image<u8, 1>) -> Result<(), StereoError> {
            self.geom.check_size(img)?;
            // Fail HERE with typed errors: the _device methods are device-only by contract.
            // `HostResident` (not `Cuda`) so a demote-to-CPU policy keyed on `Cuda` cannot
            // mistake a routing bug for a driver failure. The dispatch layer independently
            // ordinal-checks the MAPS against the frames, so a cross-device frame is a typed
            // error there too, never a foreign-pointer launch.
            let Some(stream) = img.cuda_stream() else {
                return Err(ImageError::HostResident.into());
            };
            if stream.context().ordinal() != self.stream.context().ordinal() {
                return Err(ImageError::DeviceMismatch.into());
            }
            Ok(())
        }

        /// The LEFT view's device-resident map planes as `(x, y)` — for external device
        /// samplers (e.g. a fused rectify+disparity kernel) to consume the SAME tables this
        /// rectifier launches with, instead of re-uploading a second copy that can drift.
        /// Mirrors [`StereoRectifier::left_maps`].
        pub fn left_maps(&self) -> (&Image<f32, 1>, &Image<f32, 1>) {
            (&self.left_map.x, &self.left_map.y)
        }

        /// The RIGHT view's device-resident map planes; see [`left_maps`](Self::left_maps).
        pub fn right_maps(&self) -> (&Image<f32, 1>, &Image<f32, 1>) {
            (&self.right_map.x, &self.right_map.y)
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
            self.geom.bf()
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
) -> Result<MapPair, ImageError> {
    let rect_t = rect.transpose(); // rectified-normalized -> camera-normalized
    let intrinsic = CameraIntrinsic {
        fx: cam.fx,
        fy: cam.fy,
        cx: cam.cx,
        cy: cam.cy,
    };
    let distortion = cam.distortion;

    // Same plane-pair builder the polynomial correction map uses; per-element writes are
    // pure, so its fixed-chunk rayon parallelism keeps the maps deterministic.
    let (t_x, t_y) = meshgrid_from_fn(width, height, |u, v| {
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
        Ok((du as f32, dv as f32))
    })
    .map_err(ImageError::InvalidImageShape)?;
    let size = ImageSize { width, height };
    Ok(MapPair {
        x: Image::new(size, t_x.into_vec())?,
        y: Image::new(size, t_y.into_vec())?,
    })
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

    /// A host frame into a device method must be the TYPED routing-bug error — not
    /// `Cuda`, which a demote-to-CPU policy treats as a genuine driver failure.
    #[test]
    #[cfg(feature = "cuda")]
    fn host_frame_into_device_method_is_typed() -> Result<(), Box<dyn std::error::Error>> {
        use cudarc::driver::CudaContext;
        let left = pinhole(64, 48, 40.0, 32.0, 24.0);
        let right = pinhole(64, 48, 40.0, 32.0, 24.0);
        let rect = StereoRectifier::from_calib(
            &left,
            &right,
            Mat3F64::IDENTITY,
            Vec3F64::new(-0.1, 0.0, 0.0),
        )?;
        let ctx = CudaContext::new(0)?;
        let stream = ctx.default_stream();
        let dev = rect.to_cuda(&stream)?;
        let host = dot_image(64, 48, 5, 5)?;
        let mut host_dst = Image::from_size_val(host.size(), 0u8)?;
        assert!(matches!(
            dev.rectify_left_device(&host, &mut host_dst),
            Err(StereoError::Image(ImageError::HostResident))
        ));
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
        // Deterministic pseudo-random frame (the crate's seeded-StdRng test idiom).
        use rand::{rngs::StdRng, RngExt, SeedableRng};
        let mut rng = StdRng::seed_from_u64(0x9E3779B9);
        let raw: Vec<u8> = (0..w * h).map(|_| rng.random::<u8>()).collect();
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
        let dev = rect.to_cuda(&stream)?;
        let src_dev = img.to_cuda(&stream)?;
        let mut dst_l = Image::zeros_cuda(size, &stream)?;
        let mut dst_r = Image::zeros_cuda(size, &stream)?;
        dev.rectify_left_device(&src_dev, &mut dst_l)?;
        dev.rectify_right_device(&src_dev, &mut dst_r)?;
        // to_host_into synchronizes the images' stream, so the bytes are final on return.
        let mut gpu_l = vec![0u8; w * h];
        let mut gpu_r = vec![0u8; w * h];
        dst_l.to_host_into(&mut gpu_l)?;
        dst_r.to_host_into(&mut gpu_r)?;

        assert_eq!(cpu_l.as_slice(), gpu_l.as_slice(), "left bytes diverge");
        assert_eq!(cpu_r.as_slice(), gpu_r.as_slice(), "right bytes diverge");
        Ok(())
    }
}
