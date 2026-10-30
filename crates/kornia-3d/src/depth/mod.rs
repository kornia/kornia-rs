//! Depth images to 3D: [`DepthImage`] and its unprojection to a point cloud.
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
//! All arithmetic is `f32`.

mod kernels;

use kornia_image::{Image, ImageError, ImageSize};

use crate::camera::PinholeCamera;
use crate::pointcloud::PointCloud;

/// Errors of the depth image ops.
#[derive(Debug, thiserror::Error)]
pub enum DepthError {
    /// An image operand is invalid (size mismatch, residency).
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

    /// An image is too large for the index types (`i32` coordinates, `u32`
    /// pixel indices).
    #[error("image {width}x{height} is too large")]
    ImageTooLarge {
        /// Image width.
        width: usize,
        /// Image height.
        height: usize,
    },
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
}

impl DepthSample for f32 {
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
}

/// A single-channel depth image and its `meter`, the number of depth units
/// per metre.
///
/// The `meter` follows rerun's `DepthImage` archetype
/// (<https://rerun.io/docs/reference/types/archetypes/depth_image>):
/// `1000.0` for millimetre `u16` depth, `1.0` for an `f32` image in metres.
/// It is validated once by the constructors, so the ops never re-check it.
/// The image may be host- or device-resident: the host ops reject a device
/// image, the CUDA ops a host one.
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
    ///   device-resident image.
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
}

/// Host paths read images as slices; device-resident images are rejected.
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
