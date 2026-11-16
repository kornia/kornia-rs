//! Range sensors: pinhole depth cameras and spherical LiDARs, their host
//! constants, and observations.

use std::f64::consts::{FRAC_PI_2, TAU};

use kornia_image::{Image, ImageError, ImageSize};
use kornia_oxide_kernels::tsdf_core as tc;

use super::{DepthRange, StdOps, TsdfError};
use crate::camera::PinholeCamera;
use crate::depth::DepthImage;
use crate::pose::Pose3d;

/// Largest image side: pixel coordinates stay exact in `f32`.
const MAX_SIDE: usize = 1 << 24;
/// Largest pixel count: the device computes `w * h` and strides its grid
/// loops in `u32`, so it stays below `2^31` (no wrap, `i += stride` ends).
const MAX_PIXELS: u64 = (u32::MAX / 2) as u64;
/// Relative tolerance of a wrapping azimuth grid's full turn.
const WRAP_TOLERANCE: f64 = 1e-9;

/// Rejects sizes whose pixel coordinates or indices would not be exact.
pub(crate) fn check_image_size(size: ImageSize) -> Result<(), TsdfError> {
    let ok = size.width >= 1
        && size.height >= 1
        && size.width <= MAX_SIDE
        && size.height <= MAX_SIDE
        && (size.width as u64) * (size.height as u64) <= MAX_PIXELS;
    if ok {
        Ok(())
    } else {
        Err(TsdfError::InvalidParameter {
            field: "size",
            value: (size.width as f64) * (size.height as f64),
        })
    }
}

/// `f32` pinhole constants `[fx, fy, cx, cy, inv_fx, inv_fy]` of an
/// undistorted camera.
pub(crate) fn pinhole_constants(camera: &PinholeCamera) -> Result<[f32; 6], TsdfError> {
    let (fx, fy, cx, cy) = (camera.fx, camera.fy, camera.cx, camera.cy);
    let bad = TsdfError::InvalidIntrinsics { fx, fy, cx, cy };
    if !(fx > 0.0
        && fy > 0.0
        && fx.is_finite()
        && fy.is_finite()
        && cx.is_finite()
        && cy.is_finite())
    {
        return Err(bad);
    }
    let k = [
        fx as f32,
        fy as f32,
        cx as f32,
        cy as f32,
        (1.0 / fx) as f32,
        (1.0 / fy) as f32,
    ];
    if !k.iter().all(|v| v.is_finite()) || k[4] == 0.0 || k[5] == 0.0 {
        return Err(bad);
    }
    if [camera.k1, camera.k2, camera.p1, camera.p2]
        .iter()
        .any(|&d| d != 0.0)
    {
        return Err(TsdfError::DistortedCamera);
    }
    Ok(k)
}

/// A pinhole depth camera: intrinsics (no distortion), image size and the
/// kept depth range.
#[derive(Debug, Clone)]
pub struct PinholeSensor {
    camera: PinholeCamera,
    size: ImageSize,
    range: DepthRange,
    k: [f32; 6],
}

impl PinholeSensor {
    /// Validates a pinhole sensor.
    ///
    /// # Arguments
    ///
    /// * `camera` - Intrinsics; distortion must be zero.
    /// * `size` - Depth (and colour) image size, sides in `[1, 2^24]`.
    /// * `range` - Kept depth interval (camera z, metres).
    ///
    /// # Returns
    ///
    /// The sensor.
    ///
    /// # Errors
    ///
    /// [`TsdfError::InvalidIntrinsics`], [`TsdfError::DistortedCamera`] or
    /// [`TsdfError::InvalidParameter`] (size, range).
    pub fn new(
        camera: &PinholeCamera,
        size: ImageSize,
        range: DepthRange,
    ) -> Result<Self, TsdfError> {
        let k = pinhole_constants(camera)?;
        check_image_size(size)?;
        range.validate()?;
        Ok(Self {
            camera: camera.clone(),
            size,
            range,
            k,
        })
    }

    /// Image size.
    pub fn size(&self) -> ImageSize {
        self.size
    }

    /// Intrinsics.
    pub fn camera(&self) -> &PinholeCamera {
        &self.camera
    }

    /// Kept depth interval.
    pub fn range(&self) -> DepthRange {
        self.range
    }

    pub(crate) fn params(
        &self,
        sensor_from_volume: tc::Rigid,
        volume_from_sensor: tc::Rigid,
        mpu: f32,
    ) -> tc::PinholeParams {
        let [fx, fy, cx, cy, inv_fx, inv_fy] = self.k;
        tc::PinholeParams {
            sensor_from_volume,
            volume_from_sensor,
            fx,
            fy,
            cx,
            cy,
            inv_fx,
            inv_fy,
            width: self.size.width as u32,
            height: self.size.height as u32,
            width_f: self.size.width as f32,
            height_f: self.size.height as f32,
            mpu,
            range_min: self.range.min,
            range_max: self.range.max,
        }
    }
}

/// Column layout of a spherical range image (radians).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct AzimuthGrid {
    /// Azimuth of column 0's centre, `|start| <= 2 pi`.
    pub start: f64,
    /// Signed column spacing (negative for clockwise spinners), non-zero.
    pub step: f64,
    /// The columns cover a full turn (`|W * step| = 2 pi`) and wrap.
    pub wrap: bool,
}

/// Precomputed `f32` LiDAR tables (see [`SphericalLidar::from_tables`]).
#[derive(Debug, Clone, Copy)]
pub struct LidarTables<'a> {
    /// `sin` of the lower field-of-view limit.
    pub fov_low: f32,
    /// `sin` of the upper field-of-view limit.
    pub fov_high: f32,
    /// `H - 1` strictly ascending `sin` of the boundaries between beams.
    pub row_boundaries: &'a [f32],
    /// `cos` of each row's elevation (row 0 at the top).
    pub cos_el: &'a [f32],
    /// `sin` of each row's elevation.
    pub sin_el: &'a [f32],
    /// `cos` of each column's azimuth.
    pub cos_az: &'a [f32],
    /// `sin` of each column's azimuth.
    pub sin_az: &'a [f32],
    /// `1 / step`.
    pub inv_az_step: f32,
    /// `-start / step`.
    pub az_offset: f32,
}

/// A spherical range sensor (spinning LiDAR): `W` columns of azimuth by
/// `H` beams of elevation, row 0 at the top. Its image is a
/// `DepthImage` of Euclidean range.
///
/// The projection is trig-free per point: the row is a search of `z / r`
/// in the beam boundaries' `sin`, the column is [`atan2_det`].
#[derive(Debug, Clone)]
pub struct SphericalLidar {
    size: ImageSize,
    azimuth: AzimuthGrid,
    range: DepthRange,
    fov_low: f32,
    fov_high: f32,
    bnd: Vec<f32>,
    cos_el: Vec<f32>,
    sin_el: Vec<f32>,
    cos_az: Vec<f32>,
    sin_az: Vec<f32>,
    inv_az_step: f32,
    az_offset: f32,
}

impl SphericalLidar {
    /// Builds a sensor from its beam elevations.
    ///
    /// # Arguments
    ///
    /// * `size` - `W` columns by `H` rows, both `>= 2`.
    /// * `azimuth` - Column layout.
    /// * `beam_elevations` - `H` elevations (radians), row 0 first, strictly
    ///   decreasing, in `[-pi/2, pi/2]`.
    /// * `range` - Kept range interval (metres).
    ///
    /// # Returns
    ///
    /// The sensor; its tables are computed in `f64` with `libm` and rounded
    /// once, so they are identical on every host.
    ///
    /// # Errors
    ///
    /// [`TsdfError::InvalidBeamTable`], [`TsdfError::InvalidAzimuthGrid`] or
    /// [`TsdfError::InvalidParameter`] (size, range).
    pub fn new(
        size: ImageSize,
        azimuth: AzimuthGrid,
        beam_elevations: &[f64],
        range: DepthRange,
    ) -> Result<Self, TsdfError> {
        check_lidar_size(size)?;
        check_azimuth(&azimuth, size.width)?;
        range.validate()?;
        let h = size.height;
        if beam_elevations.len() != h {
            return Err(TsdfError::InvalidBeamTable("one elevation per row"));
        }
        if !beam_elevations
            .iter()
            .all(|e| e.is_finite() && (-FRAC_PI_2..=FRAC_PI_2).contains(e))
        {
            return Err(TsdfError::InvalidBeamTable(
                "elevation outside [-pi/2, pi/2]",
            ));
        }
        if !beam_elevations.windows(2).all(|w| w[0] > w[1]) {
            return Err(TsdfError::InvalidBeamTable(
                "elevations not strictly decreasing",
            ));
        }
        // ascending elevations e_k = theta_{H-1-k}
        let e = |k: usize| beam_elevations[h - 1 - k];
        let bnd: Vec<f32> = (0..h - 1)
            .map(|j| libm::sin((e(j) + e(j + 1)) / 2.0) as f32)
            .collect();
        let lo_el = e(0) - (e(1) - e(0)) / 2.0;
        let hi_el = e(h - 1) + (e(h - 1) - e(h - 2)) / 2.0;
        let fov_low = libm::sin(if lo_el > -FRAC_PI_2 {
            lo_el
        } else {
            -FRAC_PI_2
        }) as f32;
        let fov_high = libm::sin(if hi_el < FRAC_PI_2 { hi_el } else { FRAC_PI_2 }) as f32;
        check_boundaries(fov_low, &bnd, fov_high)?;
        let cos_el = beam_elevations
            .iter()
            .map(|&t| libm::cos(t) as f32)
            .collect();
        let sin_el = beam_elevations
            .iter()
            .map(|&t| libm::sin(t) as f32)
            .collect();
        let az = |c: usize| azimuth.start + (c as f64) * azimuth.step;
        let cos_az = (0..size.width).map(|c| libm::cos(az(c)) as f32).collect();
        let sin_az = (0..size.width).map(|c| libm::sin(az(c)) as f32).collect();
        Ok(Self {
            size,
            azimuth,
            range,
            fov_low,
            fov_high,
            bnd,
            cos_el,
            sin_el,
            cos_az,
            sin_az,
            inv_az_step: (1.0 / azimuth.step) as f32,
            az_offset: (-azimuth.start / azimuth.step) as f32,
        })
    }

    /// Builds a sensor with `H` beams evenly spaced from `top` to `bottom`.
    ///
    /// # Arguments
    ///
    /// * `size` - `W` columns by `H` rows, both `>= 2`.
    /// * `azimuth` - Column layout.
    /// * `top` - Elevation of row 0 (radians), `> bottom`.
    /// * `bottom` - Elevation of row `H - 1` (radians).
    /// * `range` - Kept range interval (metres).
    ///
    /// # Returns
    ///
    /// The sensor (see [`SphericalLidar::new`]).
    ///
    /// # Errors
    ///
    /// As [`SphericalLidar::new`].
    ///
    /// # Example
    ///
    /// ```
    /// use kornia_3d::tsdf::{AzimuthGrid, DepthRange, SphericalLidar};
    /// use kornia_image::ImageSize;
    /// let w = 1024;
    /// let az = AzimuthGrid { start: 0.0, step: std::f64::consts::TAU / w as f64, wrap: true };
    /// let size = ImageSize { width: w, height: 32 };
    /// let lidar = SphericalLidar::uniform(size, az, 0.26, -0.26, DepthRange::new(0.3, 50.0)?)?;
    /// assert_eq!(lidar.size(), size);
    /// # Ok::<(), kornia_3d::tsdf::TsdfError>(())
    /// ```
    pub fn uniform(
        size: ImageSize,
        azimuth: AzimuthGrid,
        top: f64,
        bottom: f64,
        range: DepthRange,
    ) -> Result<Self, TsdfError> {
        check_lidar_size(size)?;
        let n = (size.height - 1) as f64;
        let el: Vec<f64> = (0..size.height)
            .map(|r| top + (bottom - top) * (r as f64) / n)
            .collect();
        Self::new(size, azimuth, &el, range)
    }

    /// Builds a sensor from precomputed `f32` tables (e.g. a vendor
    /// calibration), validated as finite, `|v| <= 1` and strictly ordered.
    ///
    /// # Arguments
    ///
    /// * `size` - `W` columns by `H` rows, both `>= 2`.
    /// * `azimuth` - Column layout (validated as in [`SphericalLidar::new`]).
    /// * `tables` - The tables; lengths `H - 1`, `H`, `H`, `W`, `W`.
    /// * `range` - Kept range interval (metres).
    ///
    /// # Returns
    ///
    /// The sensor.
    ///
    /// # Errors
    ///
    /// [`TsdfError::InvalidBeamTable`], [`TsdfError::InvalidAzimuthGrid`] or
    /// [`TsdfError::InvalidParameter`].
    pub fn from_tables(
        size: ImageSize,
        azimuth: AzimuthGrid,
        tables: &LidarTables<'_>,
        range: DepthRange,
    ) -> Result<Self, TsdfError> {
        check_lidar_size(size)?;
        check_azimuth(&azimuth, size.width)?;
        range.validate()?;
        let (w, h) = (size.width, size.height);
        let t = tables;
        if t.row_boundaries.len() != h - 1
            || t.cos_el.len() != h
            || t.sin_el.len() != h
            || t.cos_az.len() != w
            || t.sin_az.len() != w
        {
            return Err(TsdfError::InvalidBeamTable("table length"));
        }
        let unit = |v: &f32| v.is_finite() && v.abs() <= 1.0;
        let all_unit = [t.row_boundaries, t.cos_el, t.sin_el, t.cos_az, t.sin_az]
            .iter()
            .all(|s| s.iter().all(unit));
        if !(all_unit && unit(&t.fov_low) && unit(&t.fov_high)) {
            return Err(TsdfError::InvalidBeamTable(
                "table value not finite in [-1, 1]",
            ));
        }
        check_boundaries(t.fov_low, t.row_boundaries, t.fov_high)?;
        if !(t.inv_az_step.is_finite() && t.inv_az_step != 0.0 && t.az_offset.is_finite()) {
            return Err(TsdfError::InvalidAzimuthGrid {
                start: azimuth.start,
                step: azimuth.step,
                columns: w,
            });
        }
        Ok(Self {
            size,
            azimuth,
            range,
            fov_low: t.fov_low,
            fov_high: t.fov_high,
            bnd: t.row_boundaries.to_vec(),
            cos_el: t.cos_el.to_vec(),
            sin_el: t.sin_el.to_vec(),
            cos_az: t.cos_az.to_vec(),
            sin_az: t.sin_az.to_vec(),
            inv_az_step: t.inv_az_step,
            az_offset: t.az_offset,
        })
    }

    /// Range image size (`W` columns by `H` rows).
    pub fn size(&self) -> ImageSize {
        self.size
    }

    /// Column layout.
    pub fn azimuth(&self) -> AzimuthGrid {
        self.azimuth
    }

    /// Kept range interval.
    pub fn range(&self) -> DepthRange {
        self.range
    }

    /// The `f32` tables.
    pub fn tables(&self) -> LidarTables<'_> {
        LidarTables {
            fov_low: self.fov_low,
            fov_high: self.fov_high,
            row_boundaries: &self.bnd,
            cos_el: &self.cos_el,
            sin_el: &self.sin_el,
            cos_az: &self.cos_az,
            sin_az: &self.sin_az,
            inv_az_step: self.inv_az_step,
            az_offset: self.az_offset,
        }
    }

    pub(crate) fn boundaries(&self) -> &[f32] {
        &self.bnd
    }

    /// Direction-table entries `(cos_el, sin_el, cos_az, sin_az)` of a pixel.
    #[inline(always)]
    pub(crate) fn beam(&self, row: usize, col: usize) -> (f32, f32, f32, f32) {
        (
            self.cos_el[row],
            self.sin_el[row],
            self.cos_az[col],
            self.sin_az[col],
        )
    }

    pub(crate) fn params(
        &self,
        sensor_from_volume: tc::Rigid,
        volume_from_sensor: tc::Rigid,
        mpu: f32,
    ) -> tc::SphericalParams {
        tc::SphericalParams {
            sensor_from_volume,
            volume_from_sensor,
            fov_low: self.fov_low,
            fov_high: self.fov_high,
            inv_az_step: self.inv_az_step,
            az_offset: self.az_offset,
            width: self.size.width as u32,
            height: self.size.height as u32,
            wrap: self.azimuth.wrap as u32,
            mpu,
            range_min: self.range.min,
            range_max: self.range.max,
        }
    }

    /// Bins sensor-frame points into a range image, keeping the nearest
    /// point per pixel; pixels without a point are `0`.
    ///
    /// Each point goes through the integration's projection: range gate
    /// `[min, max)`, vertical field of view, beam row, [`atan2_det`] column.
    /// Points failing a gate (including NaN) are dropped. The result does
    /// not depend on the point order.
    ///
    /// # Arguments
    ///
    /// * `points_sensor` - Points in the sensor frame (metres).
    /// * `dst` - Host range image of the sensor's size with `meter == 1`.
    ///
    /// # Returns
    ///
    /// `Ok(())` once `dst` holds the range image.
    ///
    /// # Errors
    ///
    /// [`TsdfError::SizeMismatch`], [`TsdfError::InvalidParameter`] (`meter`),
    /// [`TsdfError::FpEnvironment`], or [`TsdfError::Image`] for a device image.
    pub fn range_image_from_points_into(
        &self,
        points_sensor: &[[f32; 3]],
        dst: &mut DepthImage<f32>,
    ) -> Result<(), TsdfError> {
        super::check_fp_env()?;
        if dst.size() != self.size {
            return Err(TsdfError::SizeMismatch {
                expected: self.size,
                got: dst.size(),
            });
        }
        if dst.meter() != 1.0 {
            return Err(TsdfError::InvalidParameter {
                field: "meter",
                value: dst.meter() as f64,
            });
        }
        check_host(dst.image())?;
        let id = tc::Rigid {
            r: [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
            t: [0.0; 3],
        };
        let p = self.params(id, id, 1.0);
        // u32::MAX bits mark an empty pixel; ranges are positive, so their
        // bit order is their value order
        let out = dst.image_mut().as_slice_mut();
        out.fill(f32::from_bits(u32::MAX));
        for &s in points_sensor {
            let Some((r, idx)) = tc::spherical_pixel::<StdOps>(&p, &self.bnd, s) else {
                continue;
            };
            // r passed `r >= range.min`, so it is not NaN
            if r >= self.range.max {
                continue;
            }
            let cur = &mut out[idx as usize];
            if r.to_bits() < cur.to_bits() {
                *cur = r;
            }
        }
        for v in out.iter_mut() {
            if v.to_bits() == u32::MAX {
                *v = 0.0;
            }
        }
        Ok(())
    }
}

fn check_lidar_size(size: ImageSize) -> Result<(), TsdfError> {
    check_image_size(size)?;
    if size.width < 2 || size.height < 2 {
        return Err(TsdfError::InvalidParameter {
            field: "size",
            value: size.width.min(size.height) as f64,
        });
    }
    Ok(())
}

fn check_azimuth(a: &AzimuthGrid, columns: usize) -> Result<(), TsdfError> {
    let full_turn = ((columns as f64) * a.step).abs();
    let ok = a.step.is_finite()
        && a.step != 0.0
        && a.start.is_finite()
        && a.start.abs() <= TAU
        && (!a.wrap || (full_turn - TAU).abs() <= WRAP_TOLERANCE * TAU)
        && (1.0 / a.step).is_finite();
    if ok {
        Ok(())
    } else {
        Err(TsdfError::InvalidAzimuthGrid {
            start: a.start,
            step: a.step,
            columns,
        })
    }
}

fn check_boundaries(lo: f32, bnd: &[f32], hi: f32) -> Result<(), TsdfError> {
    let mut prev = lo;
    for &b in bnd.iter().chain(std::iter::once(&hi)) {
        if b.partial_cmp(&prev) != Some(std::cmp::Ordering::Greater) {
            return Err(TsdfError::InvalidBeamTable(
                "row boundaries not strictly ascending after rounding",
            ));
        }
        prev = b;
    }
    Ok(())
}

pub(crate) fn check_host<T, const C: usize>(img: &Image<T, C>) -> Result<(), TsdfError> {
    if img.storage.domain().is_host_accessible() {
        Ok(())
    } else {
        Err(ImageError::UnsupportedDevice.into())
    }
}

/// A range sensor of a rig or an observation.
#[derive(Debug, Clone)]
pub enum RangeSensor {
    /// Pinhole depth camera.
    Pinhole(PinholeSensor),
    /// Spherical LiDAR.
    Spherical(SphericalLidar),
}

impl RangeSensor {
    /// Image size of the sensor.
    pub fn size(&self) -> ImageSize {
        match self {
            Self::Pinhole(s) => s.size(),
            Self::Spherical(s) => s.size(),
        }
    }
}

impl From<PinholeSensor> for RangeSensor {
    fn from(s: PinholeSensor) -> Self {
        Self::Pinhole(s)
    }
}

impl From<SphericalLidar> for RangeSensor {
    fn from(s: SphericalLidar) -> Self {
        Self::Spherical(s)
    }
}

/// Deterministic `atan2(y, x)` in radians, `[-pi, pi]`, identical on every
/// backend: an octant reduction then one minimax polynomial
/// (`|err| <= 3.3e-7` rad). Unlike libm, `atan2(-0, x < 0) = +pi`.
///
/// # Arguments
///
/// * `y`, `x` - Finite coordinates.
///
/// # Returns
///
/// The angle of `(x, y)`; `0` for `(0, 0)`.
///
/// # Example
///
/// ```
/// use kornia_3d::tsdf::atan2_det;
/// assert!((atan2_det(1.0, 1.0) - std::f32::consts::FRAC_PI_4).abs() < 1e-6);
/// ```
pub fn atan2_det(y: f32, x: f32) -> f32 {
    tc::atan2_det::<StdOps>(y, x)
}

/// A borrowed depth or range image of either sample type.
#[derive(Clone, Copy)]
pub enum DepthRef<'a> {
    /// Integer units (`meter` per metre), `0` = missing.
    U16(&'a DepthImage<u16>),
    /// Floating point, non-finite or `<= 0` = missing.
    F32(&'a DepthImage<f32>),
}

impl<'a> From<&'a DepthImage<u16>> for DepthRef<'a> {
    fn from(d: &'a DepthImage<u16>) -> Self {
        Self::U16(d)
    }
}

impl<'a> From<&'a DepthImage<f32>> for DepthRef<'a> {
    fn from(d: &'a DepthImage<f32>) -> Self {
        Self::F32(d)
    }
}

impl DepthRef<'_> {
    /// Image size.
    pub fn size(&self) -> ImageSize {
        match self {
            Self::U16(d) => d.size(),
            Self::F32(d) => d.size(),
        }
    }

    pub(crate) fn metres_per_unit(&self) -> f32 {
        match self {
            Self::U16(d) => d.metres_per_unit(),
            Self::F32(d) => d.metres_per_unit(),
        }
    }

    pub(crate) fn check_host(&self) -> Result<(), TsdfError> {
        match self {
            Self::U16(d) => check_host(d.image()),
            Self::F32(d) => check_host(d.image()),
        }
    }
}

/// One sensor's measurement in a [`super::TsdfVolume`] step.
#[derive(Clone, Copy)]
pub struct SensorObservation<'a> {
    /// The sensor.
    pub sensor: &'a RangeSensor,
    /// Sensor pose in the volume frame.
    pub volume_from_sensor: Pose3d,
    /// Depth (pinhole, camera z) or range (spherical) image of the sensor's
    /// size.
    pub depth: DepthRef<'a>,
    /// Registered colour of the depth's size (pinhole only).
    pub color: Option<&'a Image<u8, 3>>,
}

/// Rejects a sensor's image of the wrong size, colour on a range sensor
/// and device images.
pub(crate) fn check_observation(
    sensor: &RangeSensor,
    depth: &DepthRef<'_>,
    color: Option<&Image<u8, 3>>,
) -> Result<(), TsdfError> {
    let expected = sensor.size();
    if depth.size() != expected {
        return Err(TsdfError::SizeMismatch {
            expected,
            got: depth.size(),
        });
    }
    depth.check_host()?;
    if let Some(c) = color {
        if matches!(sensor, RangeSensor::Spherical(_)) {
            return Err(TsdfError::ColorOnRangeSensor);
        }
        if c.size() != expected {
            return Err(TsdfError::SizeMismatch {
                expected,
                got: c.size(),
            });
        }
        check_host(c)?;
    }
    Ok(())
}
