//! Depth images to 3D: unprojection to a point cloud.
//!
//! A depth image is a single-channel `u16` image in integer units (e.g.
//! millimetres) or an `f32` image, converted to metres by a caller-given
//! `metres_per_unit` multiplier (pass `1.0` for an `f32` image already in
//! metres).
//!
//! Validity (same rule as `kornia_imgproc::depth`): a `u16` sample is a
//! measurement iff it is non-zero; an `f32` sample iff it is finite and
//! positive, so `0`, negative values, `±inf` and `NaN` are missing. Missing
//! samples are skipped. The scale must keep valid samples representable:
//! for `u16`, `65535 · metres_per_unit` must be finite (checked), so every
//! non-zero sample has a finite positive value in metres; for `f32`, a
//! valid sample whose value in metres overflows to infinity or underflows
//! to `0` (only possible when `metres_per_unit != 1`) has no
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

use kornia_image::{Image, ImageError};

use crate::camera::PinholeCamera;
use crate::pointcloud::PointCloud;

/// Errors of the depth unprojection ops.
#[derive(Debug, thiserror::Error)]
pub enum DepthError {
    /// An image operand is invalid (size mismatch, residency).
    #[error(transparent)]
    Image(#[from] ImageError),

    /// `metres_per_unit` is not finite and positive, or, for `u16` depth,
    /// `65535 · metres_per_unit` overflows `f32`. Subnormal values are
    /// accepted.
    #[error("metres_per_unit must be finite and positive, got {0}")]
    InvalidDepthScale(f32),

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
    #[doc(hidden)]
    fn metres(self, metres_per_unit: f32) -> f32;

    /// Whether `metres_per_unit` is accepted for this dtype (see
    /// [`DepthError::InvalidDepthScale`]).
    #[doc(hidden)]
    fn scale_is_valid(metres_per_unit: f32) -> bool;

    /// Whether the sample is kept at an accepted `metres_per_unit`: the
    /// validity rule of the module docs, in the cheapest exact form.
    #[doc(hidden)]
    fn is_valid_at(self, metres_per_unit: f32) -> bool;
}

impl DepthSample for u16 {
    #[inline(always)]
    fn metres(self, metres_per_unit: f32) -> f32 {
        f32::from(self) * metres_per_unit
    }

    /// Finite and positive, and `65535 * s` finite: for `v >= 1` rounding
    /// is monotone, so `s <= v * s <= 65535 * s` and every non-zero sample
    /// has a finite positive value in metres.
    #[inline(always)]
    fn scale_is_valid(metres_per_unit: f32) -> bool {
        metres_per_unit.is_finite()
            && metres_per_unit > 0.0
            && (f32::from(u16::MAX) * metres_per_unit).is_finite()
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
    fn scale_is_valid(metres_per_unit: f32) -> bool {
        metres_per_unit.is_finite() && metres_per_unit > 0.0
    }

    /// The sample rule on `self · s`: for a finite positive `s` the product
    /// is finite and positive iff `self` is and the product neither
    /// overflows nor underflows to `0`.
    #[inline(always)]
    fn is_valid_at(self, metres_per_unit: f32) -> bool {
        kernels::is_valid_depth_f32(self.metres(metres_per_unit))
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

/// Unproject a depth image into a point cloud in the camera frame.
///
/// Every kept pixel `(x, y)` with depth `z = depth · metres_per_unit`
/// becomes the point `((x - cx) / fx · z, (y - cy) / fy · z, z)` in metres.
/// Missing samples and pixels whose `mask` value is `0` are skipped; a `u16`
/// sample is valid iff non-zero, an `f32` sample iff finite and positive
/// (same rule as `kornia_imgproc::depth`; see the module docs for samples
/// whose value in metres is not representable). Points are in row-major
/// pixel order.
///
/// Reference: Open3D `geometry::PointCloud::create_from_depth_image`.
/// Departures: depth is scaled by a multiplier `metres_per_unit` (Open3D
/// divides by `depth_scale`, so `metres_per_unit = 1 / depth_scale`); there
/// is no `depth_trunc` or `stride` (mask the far pixels instead); the
/// arithmetic is `f32`, widened to the `f64` [`PointCloud`].
///
/// # Arguments
///
/// * `depth` - Single-channel `u16` or `f32` depth image.
/// * `metres_per_unit` - Metres per depth unit (e.g. `0.001` for
///   millimetres, `1.0` for metres); see
///   [`DepthError::InvalidDepthScale`] for the accepted values.
/// * `camera` - Pinhole intrinsics of the depth image (distortion ignored).
/// * `mask` - Optional `u8` mask of the same size; only pixels with a
///   non-zero mask value are unprojected.
///
/// # Returns
///
/// A [`PointCloud`] without colors or normals.
///
/// # Errors
///
/// See [`unproject_depth_into`].
///
/// # Example
///
/// ```rust
/// use kornia_3d::camera::PinholeCamera;
/// use kornia_3d::depth::unproject_depth;
/// use kornia_image::{Image, ImageSize};
///
/// let camera = PinholeCamera {
///     fx: 2.0, fy: 2.0, cx: 1.0, cy: 1.0,
///     k1: 0.0, k2: 0.0, p1: 0.0, p2: 0.0,
/// };
/// let size = ImageSize { width: 2, height: 2 };
/// // Millimetres; one pixel is missing.
/// let depth = Image::<u16, 1>::new(size, vec![2000, 0, 1000, 1000])?;
/// let cloud = unproject_depth(&depth, 0.001, &camera, None)?;
/// assert_eq!(cloud.len(), 3);
/// // Pixel (0, 0) at 2 m: ((0 - 1) / 2 * 2, (0 - 1) / 2 * 2, 2).
/// assert_eq!(cloud.points()[0], [-1.0, -1.0, 2.0]);
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
pub fn unproject_depth<T: DepthSample>(
    depth: &Image<T, 1>,
    metres_per_unit: f32,
    camera: &PinholeCamera,
    mask: Option<&Image<u8, 1>>,
) -> Result<PointCloud, DepthError> {
    let mut points = Vec::new();
    unproject_depth_into(depth, metres_per_unit, camera, mask, &mut points, None)?;
    Ok(PointCloud::new(points, None, None))
}

/// [`unproject_depth`] into caller-owned buffers, reused across calls.
///
/// `points` is cleared and refilled; its allocation is kept, so a buffer
/// sized for one frame serves the next without reallocating. With
/// `pixel_index`, it is cleared and receives the linear index `y · width + x`
/// of each point's source pixel, in the same order. Wrap the result with
/// [`PointCloud::new`] when a [`PointCloud`] is needed. Validity is that of
/// [`unproject_depth`]: `u16` non-zero, `f32` finite and positive (same rule
/// as `kornia_imgproc::depth`).
///
/// # Arguments
///
/// * `depth` - Single-channel `u16` or `f32` depth image.
/// * `metres_per_unit` - Metres per depth unit.
/// * `camera` - Pinhole intrinsics of the depth image (distortion ignored).
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
/// * [`DepthError::InvalidDepthScale`] if `metres_per_unit` is not finite
///   and positive, or, for `u16` depth, `65535 · metres_per_unit`
///   overflows `f32` (subnormal values are accepted).
/// * [`DepthError::InvalidIntrinsics`] for a zero or non-finite focal
///   length or a non-finite principal point.
/// * [`DepthError::Image`]`(`[`ImageError::InvalidImageSize`]`)` if the mask
///   size differs from the depth size.
/// * [`DepthError::Image`]`(`[`ImageError::UnsupportedDevice`]`)` for a
///   device-resident image.
/// * [`DepthError::ImageTooLarge`] if a side does not fit `i32`, or, with
///   `pixel_index`, the pixel count exceeds `2^32`.
///
/// # Example
///
/// ```rust
/// use kornia_3d::camera::PinholeCamera;
/// use kornia_3d::depth::unproject_depth_into;
/// use kornia_image::{Image, ImageSize};
///
/// let camera = PinholeCamera {
///     fx: 1.0, fy: 1.0, cx: 0.0, cy: 0.0,
///     k1: 0.0, k2: 0.0, p1: 0.0, p2: 0.0,
/// };
/// let size = ImageSize { width: 3, height: 1 };
/// let depth = Image::<f32, 1>::new(size, vec![1.0, f32::NAN, 2.0])?;
/// let mask = Image::<u8, 1>::new(size, vec![255, 255, 0])?;
/// let mut points = Vec::new();
/// let mut index = Vec::new();
/// unproject_depth_into(&depth, 1.0, &camera, Some(&mask), &mut points, Some(&mut index))?;
/// assert_eq!(points, vec![[0.0, 0.0, 1.0]]);
/// assert_eq!(index, vec![0]);
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
pub fn unproject_depth_into<T: DepthSample>(
    depth: &Image<T, 1>,
    metres_per_unit: f32,
    camera: &PinholeCamera,
    mask: Option<&Image<u8, 1>>,
    points: &mut Vec<[f64; 3]>,
    pixel_index: Option<&mut Vec<u32>>,
) -> Result<(), DepthError> {
    check_host(depth)?;
    if let Some(m) = mask {
        check_host(m)?;
        check_same_size(depth, m)?;
    }
    let p = kernels::unproject_params::<T>(camera, metres_per_unit, depth.width(), depth.height())?;
    if pixel_index.is_some() {
        kernels::check_pixel_index(depth.width(), depth.height())?;
    }
    kernels::unproject_into(
        depth.as_slice(),
        mask.map(|m| m.as_slice()),
        &p,
        points,
        pixel_index,
    );
    Ok(())
}

#[cfg(test)]
mod tests;
