//! Depth images to 3D: [`DepthImage`], its unprojection to a point cloud,
//! and its registration (reprojection) into a second pinhole camera.
//!
//! A [`DepthImage`] is a single-channel `u16` image in integer units (e.g.
//! millimetres) or an `f32` image, plus its `meter`: the number of depth
//! units per metre (`1000.0` for millimetre `u16` depth, `1.0` for an `f32`
//! image already in metres). This is the `meter` convention of rerun's
//! `DepthImage` archetype
//! (<https://rerun.io/docs/reference/types/archetypes/depth_image>); its
//! display-only fields (`colormap`, `depth_range`, `point_fill_ratio`) are
//! not carried. A sample `d` is `d / meter` metres; the kernels compute it
//! as `d · (1 / meter)`, with `1 / meter` rounded to `f32` once per call
//! (exact for power-of-two `meter`s; `1f32 / 1000.0 == 0.001f32`).
//!
//! Validity (same rule as `kornia_imgproc::depth`): a `u16` sample is a
//! measurement iff it is non-zero; an `f32` sample iff it is finite and
//! positive, so `0`, negative values, `±inf` and `NaN` are missing. Missing
//! samples are skipped. The `meter` must keep valid samples representable
//! (see [`DepthError::InvalidMeter`]): for `u16`, `65535 · (1 / meter)` must
//! be finite, so every non-zero sample has a finite positive value in
//! metres; for `f32`, a valid sample whose value in metres overflows to
//! infinity or underflows to `0` (only possible when `meter != 1`) has no
//! representable point and is skipped too.
//!
//! Pixel coordinates follow [`PinholeCamera`]: the centre of pixel
//! `(x, y)` is at `(x, y)`, and a camera-frame point `(X, Y, Z)` projects to
//! `(fx·X/Z + cx, fy·Y/Z + cy)`. Only `fx, fy, cx, cy` are used: the
//! distortion coefficients are ignored, so the depth image must already be
//! undistorted (as depth sensors and Open3D's `PinholeCameraIntrinsic`
//! assume).
//!
//! All arithmetic is `f32`, on the host and on the device (feature `cuda`,
//! `CudaDepthProjector`); the device kernels mirror the host expressions
//! and produce identical results.

mod kernels;

#[cfg(feature = "cuda")]
mod cuda;

use kornia_image::{Image, ImageError, ImageSize};

use crate::camera::PinholeCamera;
use crate::pointcloud::PointCloud;
use crate::pose::Pose3d;

pub use kernels::MAX_SPLAT_RADIUS;

#[cfg(feature = "cuda")]
pub use cuda::{CudaDepthProjector, CudaPointCloud};

/// Errors of the depth image ops.
#[derive(Debug, thiserror::Error)]
pub enum DepthError {
    /// An image operand is invalid (size mismatch, residency, CUDA failure).
    #[error(transparent)]
    Image(#[from] ImageError),

    /// `meter` (depth units per metre) is not finite and positive, its
    /// reciprocal overflows `f32`, or, for `u16` depth,
    /// `65535 · (1 / meter)` overflows `f32`.
    #[error("meter must be finite and positive with a finite reciprocal, got {0}")]
    InvalidMeter(f32),

    /// A camera's focal lengths are zero or not finite, or its principal
    /// point is not finite.
    #[error("invalid pinhole intrinsics fx={fx} fy={fy} cx={cx} cy={cy}")]
    InvalidIntrinsics {
        /// Focal length x.
        fx: f64,
        /// Focal length y.
        fy: f64,
        /// Principal point x.
        cx: f64,
        /// Principal point y.
        cy: f64,
    },

    /// The `dst_from_src` transform has a non-finite entry.
    #[error("dst_from_src has a non-finite rotation or translation entry")]
    InvalidTransform,

    /// The splat radius exceeds [`MAX_SPLAT_RADIUS`].
    #[error("splat radius {radius} exceeds the maximum {max}")]
    SplatRadiusTooLarge {
        /// Requested radius.
        radius: usize,
        /// Largest accepted radius.
        max: usize,
    },

    /// An image is too large for the index types (`i32` coordinates, `u32`
    /// pixel indices and point counts).
    #[error("image {width}x{height} is too large")]
    ImageTooLarge {
        /// Image width.
        width: usize,
        /// Image height.
        height: usize,
    },

    /// A device point buffer cannot hold one point per depth pixel
    /// (feature `cuda`).
    #[error("point buffer capacity {capacity} < {needed} depth pixels")]
    CapacityTooSmall {
        /// Buffer capacity in points.
        capacity: usize,
        /// Depth pixels of the call.
        needed: usize,
    },

    /// A device buffer belongs to a different stream than the projector
    /// using it (feature `cuda`).
    #[error("device buffer was allocated on a different stream than the projector")]
    StreamMismatch,
}

mod sealed {
    pub trait Sealed {}
    impl Sealed for u16 {}
    impl Sealed for f32 {}
}

/// Depth sample types: `u16` (integer units) and `f32`. Sealed.
///
/// Distinct from `kornia_imgproc::depth::DepthPixel`, whose hidden methods
/// are the image-filter kernels of that crate.
pub trait DepthSample: sealed::Sealed + Copy + Send + Sync + 'static {
    #[doc(hidden)]
    const KERNEL_INDEX: usize;

    /// The sample in metres, given `metres_per_unit = 1 / meter`.
    #[doc(hidden)]
    fn metres(self, metres_per_unit: f32) -> f32;

    /// Whether `meter` is accepted for this dtype (see
    /// [`DepthError::InvalidMeter`]).
    #[doc(hidden)]
    fn meter_is_valid(meter: f32) -> bool;

    /// Whether the sample is kept at the `metres_per_unit` of an accepted
    /// `meter`: the validity rule of the module docs, in the cheapest exact
    /// form.
    #[doc(hidden)]
    fn is_valid_at(self, metres_per_unit: f32) -> bool;

    /// The host registration kernel of this dtype (see
    /// [`DepthImage::register_into`]).
    #[doc(hidden)]
    fn register_impl(
        src: &DepthImage<Self>,
        camera: &PinholeCamera,
        dst_from_src: &Pose3d,
        dst_camera: &PinholeCamera,
        splat_radius: usize,
        dst: &mut DepthImage<Self>,
    ) -> Result<(), DepthError>;
}

/// `1 / meter` in `f32`: the multiplier the kernels apply to a sample.
#[inline(always)]
fn metres_per_unit(meter: f32) -> f32 {
    1.0 / meter
}

/// `meter` is finite and positive and its reciprocal is finite. The
/// reciprocal of a finite positive `f32` is never `0` (it is at least
/// `1 / f32::MAX`, a subnormal), so it is then finite and positive.
#[inline(always)]
fn meter_is_finite_positive(meter: f32) -> bool {
    meter.is_finite() && meter > 0.0 && metres_per_unit(meter).is_finite()
}

impl DepthSample for u16 {
    const KERNEL_INDEX: usize = kernels::KIND_U16;

    #[inline(always)]
    fn metres(self, metres_per_unit: f32) -> f32 {
        f32::from(self) * metres_per_unit
    }

    /// [`meter_is_finite_positive`] and `65535 · s` finite for
    /// `s = 1 / meter`: for `v >= 1` rounding is monotone, so
    /// `s <= v · s <= 65535 · s` and every non-zero sample has a finite
    /// positive value in metres.
    #[inline(always)]
    fn meter_is_valid(meter: f32) -> bool {
        meter_is_finite_positive(meter)
            && (f32::from(u16::MAX) * metres_per_unit(meter)).is_finite()
    }

    #[inline(always)]
    fn is_valid_at(self, _metres_per_unit: f32) -> bool {
        kernels::is_valid_depth_u16(self)
    }

    fn register_impl(
        src: &DepthImage<Self>,
        camera: &PinholeCamera,
        dst_from_src: &Pose3d,
        dst_camera: &PinholeCamera,
        splat_radius: usize,
        dst: &mut DepthImage<Self>,
    ) -> Result<(), DepthError> {
        let p = register_checks(src, camera, dst_from_src, dst_camera, splat_radius, dst)?;
        kernels::register_u16(src.image.as_slice(), dst.image.as_slice_mut(), &p);
        Ok(())
    }
}

impl DepthSample for f32 {
    const KERNEL_INDEX: usize = kernels::KIND_F32;

    #[inline(always)]
    fn metres(self, metres_per_unit: f32) -> f32 {
        self * metres_per_unit
    }

    #[inline(always)]
    fn meter_is_valid(meter: f32) -> bool {
        meter_is_finite_positive(meter)
    }

    /// The sample rule on `self · s`: for a finite positive `s` the product
    /// is finite and positive iff `self` is and the product neither
    /// overflows nor underflows to `0`.
    #[inline(always)]
    fn is_valid_at(self, metres_per_unit: f32) -> bool {
        kernels::is_valid_depth_f32(self.metres(metres_per_unit))
    }

    fn register_impl(
        src: &DepthImage<Self>,
        camera: &PinholeCamera,
        dst_from_src: &Pose3d,
        dst_camera: &PinholeCamera,
        splat_radius: usize,
        dst: &mut DepthImage<Self>,
    ) -> Result<(), DepthError> {
        let p = register_checks(src, camera, dst_from_src, dst_camera, splat_radius, dst)?;
        kernels::register_f32(src.image.as_slice(), dst.image.as_slice_mut(), &p);
        Ok(())
    }
}

/// A single-channel depth image and its `meter`, the number of depth units
/// per metre.
///
/// The `meter` follows rerun's `DepthImage` archetype
/// (<https://rerun.io/docs/reference/types/archetypes/depth_image>):
/// `1000.0` for millimetre `u16` depth, `1.0` for an `f32` image in metres.
/// It is validated once by the constructors, so the ops never re-check it.
/// The image may be host- or device-resident: the host ops reject a device
/// image, the `CudaDepthProjector` ops (feature `cuda`) a host one.
#[derive(Clone)]
pub struct DepthImage<T: DepthSample> {
    image: Image<T, 1>,
    meter: f32,
}

impl<T: DepthSample> DepthImage<T> {
    /// Wraps a depth image with its `meter`.
    ///
    /// # Arguments
    ///
    /// * `image` - Single-channel `u16` or `f32` depth image (host or
    ///   device).
    /// * `meter` - Depth units per metre (e.g. `1000.0` for millimetres,
    ///   `1.0` for metres).
    ///
    /// # Returns
    ///
    /// The depth image.
    ///
    /// # Errors
    ///
    /// [`DepthError::InvalidMeter`] unless `meter` is finite, positive and
    /// has a finite `f32` reciprocal (`meter >= 1 / f32::MAX`, about
    /// `2.94e-39`), and, for `u16`, `65535 · (1 / meter)` is finite
    /// (`meter` from about `1.926e-34`). Both bounds are monotone in
    /// `meter`, so each dtype accepts the interval from its smallest
    /// `meter` up to `f32::MAX`.
    ///
    /// # Example
    ///
    /// ```rust
    /// use kornia_3d::depth::{DepthError, DepthImage};
    /// use kornia_image::{Image, ImageSize};
    ///
    /// let size = ImageSize { width: 2, height: 1 };
    /// let depth = DepthImage::new(Image::<u16, 1>::new(size, vec![1500, 0])?, 1000.0)?;
    /// assert_eq!(depth.meter(), 1000.0);
    /// let bad = DepthImage::new(Image::<u16, 1>::new(size, vec![1500, 0])?, 0.0);
    /// assert!(matches!(bad, Err(DepthError::InvalidMeter(_))));
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    pub fn new(image: Image<T, 1>, meter: f32) -> Result<Self, DepthError> {
        if !T::meter_is_valid(meter) {
            return Err(DepthError::InvalidMeter(meter));
        }
        Ok(Self { image, meter })
    }

    /// A host depth image of `size` filled with `val`.
    ///
    /// # Arguments
    ///
    /// * `size` - Image size.
    /// * `val` - Value of every pixel (`0` = missing).
    /// * `meter` - Depth units per metre.
    ///
    /// # Returns
    ///
    /// The depth image.
    ///
    /// # Errors
    ///
    /// [`DepthError::InvalidMeter`] as for [`DepthImage::new`];
    /// [`DepthError::Image`] if the image cannot be allocated.
    pub fn from_size_val(size: ImageSize, val: T, meter: f32) -> Result<Self, DepthError> {
        Self::new(Image::from_size_val(size, val)?, meter)
    }

    /// The depth pixels.
    ///
    /// # Returns
    ///
    /// The wrapped image.
    ///
    /// # Errors
    ///
    /// This method is infallible.
    pub fn image(&self) -> &Image<T, 1> {
        &self.image
    }

    /// The depth pixels, mutably; the `meter` is unchanged.
    ///
    /// # Returns
    ///
    /// The wrapped image, mutably.
    ///
    /// # Errors
    ///
    /// This method is infallible.
    pub fn image_mut(&mut self) -> &mut Image<T, 1> {
        &mut self.image
    }

    /// Unwraps the depth pixels.
    ///
    /// # Returns
    ///
    /// The wrapped image; the `meter` is dropped.
    ///
    /// # Errors
    ///
    /// This method is infallible.
    pub fn into_inner(self) -> Image<T, 1> {
        self.image
    }

    /// Depth units per metre.
    ///
    /// # Returns
    ///
    /// The `meter` given at construction.
    ///
    /// # Errors
    ///
    /// This method is infallible.
    pub fn meter(&self) -> f32 {
        self.meter
    }

    /// Image size.
    ///
    /// # Returns
    ///
    /// Width and height in pixels.
    ///
    /// # Errors
    ///
    /// This method is infallible.
    pub fn size(&self) -> ImageSize {
        self.image.size()
    }

    /// Image width.
    ///
    /// # Returns
    ///
    /// Width in pixels.
    ///
    /// # Errors
    ///
    /// This method is infallible.
    pub fn width(&self) -> usize {
        self.image.width()
    }

    /// Image height.
    ///
    /// # Returns
    ///
    /// Height in pixels.
    ///
    /// # Errors
    ///
    /// This method is infallible.
    pub fn height(&self) -> usize {
        self.image.height()
    }

    /// `1 / meter` in `f32`, the multiplier the kernels apply to a sample.
    pub(crate) fn metres_per_unit(&self) -> f32 {
        metres_per_unit(self.meter)
    }

    /// Unprojects the depth image into a point cloud in the camera frame.
    ///
    /// Every kept pixel `(x, y)` with depth `z = depth · (1 / meter)`
    /// becomes the point `((x - cx) / fx · z, (y - cy) / fy · z, z)` in
    /// metres. Missing samples and pixels whose `mask` value is `0` are
    /// skipped; a `u16` sample is valid iff non-zero, an `f32` sample iff
    /// finite and positive (same rule as `kornia_imgproc::depth`; see the
    /// module docs for samples whose value in metres is not
    /// representable). Points are in row-major pixel order.
    ///
    /// Reference: Open3D `geometry::PointCloud::create_from_depth_image`.
    /// Departures: depth is scaled by `1 / meter` rounded to `f32` (Open3D
    /// divides by its `depth_scale`, which is `meter`); there is no
    /// `depth_trunc` or `stride` (mask the far pixels instead); the
    /// arithmetic is `f32`, widened to the `f64` [`PointCloud`].
    ///
    /// # Arguments
    ///
    /// * `camera` - Pinhole intrinsics of the depth image (distortion
    ///   ignored).
    /// * `mask` - Optional `u8` mask of the same size; only pixels with a
    ///   non-zero mask value are unprojected.
    ///
    /// # Returns
    ///
    /// A [`PointCloud`] without colors or normals.
    ///
    /// # Errors
    ///
    /// See [`DepthImage::to_pointcloud_into`].
    ///
    /// # Example
    ///
    /// ```rust
    /// use kornia_3d::camera::PinholeCamera;
    /// use kornia_3d::depth::DepthImage;
    /// use kornia_image::{Image, ImageSize};
    ///
    /// let camera = PinholeCamera {
    ///     fx: 2.0, fy: 2.0, cx: 1.0, cy: 1.0,
    ///     k1: 0.0, k2: 0.0, p1: 0.0, p2: 0.0,
    /// };
    /// let size = ImageSize { width: 2, height: 2 };
    /// // Millimetres; one pixel is missing.
    /// let depth = DepthImage::new(Image::<u16, 1>::new(size, vec![2000, 0, 1000, 1000])?, 1000.0)?;
    /// let cloud = depth.to_pointcloud(&camera, None)?;
    /// assert_eq!(cloud.len(), 3);
    /// // Pixel (0, 0) at 2 m: ((0 - 1) / 2 * 2, (0 - 1) / 2 * 2, 2).
    /// assert_eq!(cloud.points()[0], [-1.0, -1.0, 2.0]);
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    pub fn to_pointcloud(
        &self,
        camera: &PinholeCamera,
        mask: Option<&Image<u8, 1>>,
    ) -> Result<PointCloud, DepthError> {
        let mut points = Vec::new();
        self.to_pointcloud_into(camera, mask, &mut points, None)?;
        Ok(PointCloud::new(points, None, None))
    }

    /// [`DepthImage::to_pointcloud`] into caller-owned buffers, reused
    /// across calls.
    ///
    /// `points` is cleared and refilled; its allocation is kept, so a buffer
    /// sized for one frame serves the next without reallocating. With
    /// `pixel_index`, it is cleared and receives the linear index
    /// `y · width + x` of each point's source pixel, in the same order. Wrap
    /// the result with [`PointCloud::new`] when a [`PointCloud`] is needed.
    /// Validity is that of [`DepthImage::to_pointcloud`].
    ///
    /// # Arguments
    ///
    /// * `camera` - Pinhole intrinsics of the depth image (distortion
    ///   ignored).
    /// * `mask` - Optional `u8` mask of the same size; non-zero = keep.
    /// * `points` - Output camera-frame points in metres, row-major order.
    /// * `pixel_index` - Optional output source-pixel index per point.
    ///
    /// # Returns
    ///
    /// `Ok(())`; `points` (and `pixel_index`) hold the result.
    ///
    /// # Errors
    ///
    /// * [`DepthError::InvalidIntrinsics`] for a zero or non-finite focal
    ///   length or a non-finite principal point.
    /// * [`DepthError::Image`]`(`[`ImageError::InvalidImageSize`]`)` if the
    ///   mask size differs from the depth size.
    /// * [`DepthError::Image`]`(`[`ImageError::UnsupportedDevice`]`)` for a
    ///   device-resident image (use `CudaDepthProjector`).
    /// * [`DepthError::ImageTooLarge`] if a side does not fit `i32`, or,
    ///   with `pixel_index`, the pixel count exceeds `2^32`.
    ///
    /// # Example
    ///
    /// ```rust
    /// use kornia_3d::camera::PinholeCamera;
    /// use kornia_3d::depth::DepthImage;
    /// use kornia_image::{Image, ImageSize};
    ///
    /// let camera = PinholeCamera {
    ///     fx: 1.0, fy: 1.0, cx: 0.0, cy: 0.0,
    ///     k1: 0.0, k2: 0.0, p1: 0.0, p2: 0.0,
    /// };
    /// let size = ImageSize { width: 3, height: 1 };
    /// let depth = DepthImage::new(Image::<f32, 1>::new(size, vec![1.0, f32::NAN, 2.0])?, 1.0)?;
    /// let mask = Image::<u8, 1>::new(size, vec![255, 255, 0])?;
    /// let mut points = Vec::new();
    /// let mut index = Vec::new();
    /// depth.to_pointcloud_into(&camera, Some(&mask), &mut points, Some(&mut index))?;
    /// assert_eq!(points, vec![[0.0, 0.0, 1.0]]);
    /// assert_eq!(index, vec![0]);
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    pub fn to_pointcloud_into(
        &self,
        camera: &PinholeCamera,
        mask: Option<&Image<u8, 1>>,
        points: &mut Vec<[f64; 3]>,
        pixel_index: Option<&mut Vec<u32>>,
    ) -> Result<(), DepthError> {
        check_host(&self.image)?;
        if let Some(m) = mask {
            check_host(m)?;
            check_same_size(&self.image, m)?;
        }
        let p = kernels::unproject_params(camera, self)?;
        if pixel_index.is_some() {
            kernels::check_pixel_index(self.width(), self.height())?;
        }
        kernels::unproject_into(
            self.image.as_slice(),
            mask.map(|m| m.as_slice()),
            &p,
            points,
            pixel_index,
        );
        Ok(())
    }

    /// Registers (reprojects) the depth image into a second pinhole camera,
    /// into a new image of `dst_size` with this image's `meter`.
    ///
    /// See [`DepthImage::register_into`] for the method; this form
    /// allocates the destination.
    ///
    /// # Arguments
    ///
    /// * `camera` - Pinhole intrinsics of this image (distortion ignored).
    /// * `dst_from_src` - Rigid transform from this camera's frame to the
    ///   destination camera's frame.
    /// * `dst_camera` - Pinhole intrinsics of the destination grid.
    /// * `splat_radius` - Splat footprint radius in destination pixels,
    ///   `0` (one pixel) to [`MAX_SPLAT_RADIUS`].
    /// * `dst_size` - Destination grid size.
    ///
    /// # Returns
    ///
    /// The registered depth image, in this image's units.
    ///
    /// # Errors
    ///
    /// See [`DepthImage::register_into`].
    ///
    /// # Example
    ///
    /// ```rust
    /// use kornia_3d::camera::PinholeCamera;
    /// use kornia_3d::depth::DepthImage;
    /// use kornia_3d::pose::Pose3d;
    /// use kornia_algebra::{Mat3F64, Vec3F64};
    /// use kornia_image::{Image, ImageSize};
    ///
    /// let camera = PinholeCamera {
    ///     fx: 100.0, fy: 100.0, cx: 2.0, cy: 0.0,
    ///     k1: 0.0, k2: 0.0, p1: 0.0, p2: 0.0,
    /// };
    /// let size = ImageSize { width: 5, height: 1 };
    /// // A wall 1 m away seen by the centre pixel only, in millimetres.
    /// let src = DepthImage::new(Image::<u16, 1>::new(size, vec![0, 0, 1000, 0, 0])?, 1000.0)?;
    /// // The destination camera sits 1 cm to the left: points move 1 px right.
    /// let dst_from_src = Pose3d::new(Mat3F64::IDENTITY, Vec3F64::new(0.01, 0.0, 0.0));
    /// let dst = src.register(&camera, &dst_from_src, &camera, 0, size)?;
    /// assert_eq!(dst.image().as_slice(), &[0, 0, 0, 1000, 0]);
    /// assert_eq!(dst.meter(), 1000.0);
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    pub fn register(
        &self,
        camera: &PinholeCamera,
        dst_from_src: &Pose3d,
        dst_camera: &PinholeCamera,
        splat_radius: usize,
        dst_size: ImageSize,
    ) -> Result<DepthImage<T>, DepthError>
    where
        T: Default,
    {
        let mut dst = Self::from_size_val(dst_size, T::default(), self.meter)?;
        self.register_into(camera, dst_from_src, dst_camera, splat_radius, &mut dst)?;
        Ok(dst)
    }

    /// Registers (reprojects) the depth image into a second pinhole camera,
    /// writing `dst` in `dst`'s own units.
    ///
    /// Each valid pixel of `self` (`u16` non-zero, `f32` finite and
    /// positive: same rule as `kornia_imgproc::depth`) is unprojected with
    /// `camera`, moved into the destination frame by `dst_from_src`
    /// (`p_dst = R · p_src + t`, metres), and projected with `dst_camera` to
    /// its nearest destination pixel (`floor(u + 0.5)`, `floor(v + 0.5)`).
    /// With `splat_radius = r` it is written to every pixel of the
    /// `(2r + 1)²` square around that pixel (clipped to the image), which
    /// closes the holes left when the destination samples the scene more
    /// densely than the source (e.g. a 2× upsample needs `r = 1`). Where
    /// several points land on one pixel, the nearest (smallest destination
    /// depth) wins: a z-buffer, so a foreground surface correctly occludes
    /// the background it moves in front of.
    ///
    /// Each image carries its own `meter` (as in rerun): `dst` gets the
    /// destination depth `z_dst` (metres) in its own units, `z_dst / s` with
    /// `s = 1 / dst.meter()` rounded to `f32`, the multiplier an
    /// unprojection of `dst` applies. `u16` values are rounded (`round`,
    /// half away from zero) and points that round to `0` or above `65534`
    /// are dropped; points whose `f32` value is not finite and positive are
    /// dropped. Pixels no point reaches are `0` (both dtypes). The division
    /// is correctly rounded and inverts the unprojection's multiplication,
    /// so an `f32` identity registration between equal `meter`s returns
    /// `(d · s) / s`: `d` itself when `s` is a power of two (e.g.
    /// `meter = 1`), otherwise `d` or a neighbour 1 ulp away while `d · s`
    /// is a normal float, since that product is already rounded. Writing
    /// `z_dst · meter` instead would round-trip far fewer samples (the
    /// rustdoc of the kernel has the numbers), as `s` is rounded.
    ///
    /// The projection restates `camera::pinhole::project_point` in `f32`
    /// rather than calling it (that function is `f64`): the host, NEON and
    /// CUDA paths evaluate the same `f32` operations in the same order,
    /// which is what makes their outputs bit-identical.
    ///
    /// The z-buffer holds an inverted depth key (`!value` for `u16`,
    /// `!bits(value)` for `f32`: positive floats order like their bit
    /// patterns) in `dst` itself, and every cell keeps its largest key,
    /// i.e. its nearest point. On the CPU the projection runs 4 lanes at a
    /// time (NEON) and the source rows are cut into bands processed in two
    /// alternating parallel phases, each band writing only its own window
    /// of destination rows with plain stores; splats that leave their
    /// window are merged with an atomic max afterwards. On CUDA every splat
    /// is an `atomicMax`. A per-cell maximum does not depend on the update
    /// order, so the result is deterministic across threads, runs and the
    /// CUDA path.
    ///
    /// Reference: ROS `depth_image_proc/register` (nearest-pixel projection
    /// with a z-buffer) and Open3D
    /// `t::geometry::PointCloud::project_to_depth_image`. Departures: ROS
    /// writes single pixels only (`r = 0`) and uses `f64` arithmetic; here
    /// an optional fixed `(2r + 1)²` square splat is added and the
    /// arithmetic is `f32`; distortion is ignored.
    ///
    /// # Arguments
    ///
    /// * `camera` - Pinhole intrinsics of this image (distortion ignored).
    /// * `dst_from_src` - Rigid transform from this camera's frame to the
    ///   destination camera's frame.
    /// * `dst_camera` - Pinhole intrinsics of the destination grid.
    /// * `splat_radius` - Splat footprint radius in destination pixels,
    ///   `0` (one pixel) to [`MAX_SPLAT_RADIUS`].
    /// * `dst` - Pre-allocated destination depth image; its size is the
    ///   destination grid, its `meter` the output units, and every pixel is
    ///   overwritten.
    ///
    /// # Returns
    ///
    /// `Ok(())`; `dst` holds the registered depth.
    ///
    /// # Errors
    ///
    /// * [`DepthError::InvalidIntrinsics`] (either camera),
    ///   [`DepthError::InvalidTransform`] for a non-finite transform.
    /// * [`DepthError::SplatRadiusTooLarge`] if `splat_radius` exceeds
    ///   [`MAX_SPLAT_RADIUS`].
    /// * [`DepthError::Image`]`(`[`ImageError::UnsupportedDevice`]`)` for a
    ///   device-resident image (use `CudaDepthProjector`).
    /// * [`DepthError::ImageTooLarge`] if a side does not fit `i32`.
    ///
    /// # Example
    ///
    /// ```rust
    /// use kornia_3d::camera::PinholeCamera;
    /// use kornia_3d::depth::DepthImage;
    /// use kornia_3d::pose::Pose3d;
    /// use kornia_image::{Image, ImageSize};
    ///
    /// let camera = PinholeCamera {
    ///     fx: 100.0, fy: 100.0, cx: 1.0, cy: 0.0,
    ///     k1: 0.0, k2: 0.0, p1: 0.0, p2: 0.0,
    /// };
    /// let size = ImageSize { width: 3, height: 1 };
    /// // Millimetres in, metres out: each image carries its own meter.
    /// let src = DepthImage::new(Image::<f32, 1>::new(size, vec![0.0, 2000.0, 0.0])?, 1000.0)?;
    /// let mut dst = DepthImage::<f32>::from_size_val(size, 0.0, 1.0)?;
    /// src.register_into(&camera, &Pose3d::IDENTITY, &camera, 0, &mut dst)?;
    /// assert_eq!(dst.image().as_slice(), &[0.0, 2.0, 0.0]);
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    pub fn register_into(
        &self,
        camera: &PinholeCamera,
        dst_from_src: &Pose3d,
        dst_camera: &PinholeCamera,
        splat_radius: usize,
        dst: &mut DepthImage<T>,
    ) -> Result<(), DepthError> {
        T::register_impl(self, camera, dst_from_src, dst_camera, splat_radius, dst)
    }
}

/// Host residency of both images, then [`kernels::register_params`].
fn register_checks<T: DepthSample>(
    src: &DepthImage<T>,
    camera: &PinholeCamera,
    dst_from_src: &Pose3d,
    dst_camera: &PinholeCamera,
    splat_radius: usize,
    dst: &DepthImage<T>,
) -> Result<kernels::RegisterParams, DepthError> {
    check_host(&src.image)?;
    check_host(&dst.image)?;
    kernels::register_params(src, camera, dst_from_src, dst_camera, splat_radius, dst)
}

/// Host paths read images as slices; device images go through
/// `CudaDepthProjector` instead.
fn check_host<T, const C: usize>(img: &Image<T, C>) -> Result<(), DepthError> {
    if img.storage.domain().is_host_accessible() {
        Ok(())
    } else {
        Err(ImageError::UnsupportedDevice.into())
    }
}

fn check_same_size<T, U>(a: &Image<T, 1>, b: &Image<U, 1>) -> Result<(), DepthError> {
    if a.size() != b.size() {
        return Err(ImageError::InvalidImageSize(a.cols(), a.rows(), b.cols(), b.rows()).into());
    }
    Ok(())
}

#[cfg(test)]
mod tests;

#[cfg(all(test, feature = "cuda"))]
mod cuda_tests;
