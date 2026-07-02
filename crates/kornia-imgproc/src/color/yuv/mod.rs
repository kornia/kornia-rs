use kornia_image::{Image, ImageError};
use rayon::prelude::*;

pub(crate) mod kernels;
use kernels::ChromaOrder;
pub use kernels::{Packed422, Planar420};

// ===== Family A: RGB ↔ YCbCr / YUV (planar 3-channel) ===============================

use crate::color::kernel_common::{check_size, sealed};

/// Compile-time dispatch to the right RGB↔{YCbCr,YUV} kernel for each pixel type.
///
/// `u8` runs the Q14 fixed-point NEON path; `f32` runs the `[0,1]` NEON path; `f64` uses
/// the portable scalar oracle. The same trait drives both YCbCr (chroma order `[Y,Cr,Cb]`)
/// and YUV (`[Y,Cb,Cr]`) via the `order` argument. Sealed.
pub trait YuvFamily: sealed::Sealed + Sized {
    #[doc(hidden)]
    fn ycc_from_rgb_impl(
        src: &Image<Self, 3>,
        dst: &mut Image<Self, 3>,
        order: ChromaOrder,
    ) -> Result<(), ImageError>;

    #[doc(hidden)]
    fn rgb_from_ycc_impl(
        src: &Image<Self, 3>,
        dst: &mut Image<Self, 3>,
        order: ChromaOrder,
    ) -> Result<(), ImageError>;
}

impl YuvFamily for u8 {
    fn ycc_from_rgb_impl(
        src: &Image<u8, 3>,
        dst: &mut Image<u8, 3>,
        order: ChromaOrder,
    ) -> Result<(), ImageError> {
        check_size(src, dst)?;
        kernels::ycc_from_rgb_u8(
            src.as_slice(),
            dst.as_slice_mut(),
            src.rows() * src.cols(),
            order,
        );
        Ok(())
    }
    fn rgb_from_ycc_impl(
        src: &Image<u8, 3>,
        dst: &mut Image<u8, 3>,
        order: ChromaOrder,
    ) -> Result<(), ImageError> {
        check_size(src, dst)?;
        kernels::rgb_from_ycc_u8(
            src.as_slice(),
            dst.as_slice_mut(),
            src.rows() * src.cols(),
            order,
        );
        Ok(())
    }
}

impl YuvFamily for f32 {
    fn ycc_from_rgb_impl(
        src: &Image<f32, 3>,
        dst: &mut Image<f32, 3>,
        order: ChromaOrder,
    ) -> Result<(), ImageError> {
        check_size(src, dst)?;
        kernels::ycc_from_rgb_f32(
            src.as_slice(),
            dst.as_slice_mut(),
            src.rows() * src.cols(),
            order,
        );
        Ok(())
    }
    fn rgb_from_ycc_impl(
        src: &Image<f32, 3>,
        dst: &mut Image<f32, 3>,
        order: ChromaOrder,
    ) -> Result<(), ImageError> {
        check_size(src, dst)?;
        kernels::rgb_from_ycc_f32(
            src.as_slice(),
            dst.as_slice_mut(),
            src.rows() * src.cols(),
            order,
        );
        Ok(())
    }
}

impl YuvFamily for f64 {
    fn ycc_from_rgb_impl(
        src: &Image<f64, 3>,
        dst: &mut Image<f64, 3>,
        order: ChromaOrder,
    ) -> Result<(), ImageError> {
        check_size(src, dst)?;
        let (yr, yg, yb, kcr, kcb) = (0.299f64, 0.587, 0.114, 0.713, 0.564);
        crate::parallel::par_iter_rows(src, dst, |s, d| {
            let (r, g, b) = (s[0], s[1], s[2]);
            let y = yr * r + yg * g + yb * b;
            let cr = (r - y) * kcr + 0.5;
            let cb = (b - y) * kcb + 0.5;
            d[0] = y;
            match order {
                ChromaOrder::YCrCb => {
                    d[1] = cr;
                    d[2] = cb;
                }
                ChromaOrder::YuvCbCr => {
                    d[1] = cb;
                    d[2] = cr;
                }
            }
        });
        Ok(())
    }
    fn rgb_from_ycc_impl(
        src: &Image<f64, 3>,
        dst: &mut Image<f64, 3>,
        order: ChromaOrder,
    ) -> Result<(), ImageError> {
        check_size(src, dst)?;
        let (yr, yg, yb, kcr, kcb) = (0.299f64, 0.587, 0.114, 0.713, 0.564);
        crate::parallel::par_iter_rows(src, dst, |s, d| {
            let y = s[0];
            let (cr, cb) = match order {
                ChromaOrder::YCrCb => (s[1], s[2]),
                ChromaOrder::YuvCbCr => (s[2], s[1]),
            };
            let r = y + (cr - 0.5) / kcr;
            let b = y + (cb - 0.5) / kcb;
            let g = (y - yr * r - yb * b) / yg;
            d[0] = r;
            d[1] = g;
            d[2] = b;
        });
        Ok(())
    }
}

/// Convert an RGB image to YCbCr (OpenCV `YCrCb`), channel order `[Y, Cr, Cb]`, full range.
///
/// Dispatches at compile time on pixel type `T`: `u8` (Q14 fixed-point), `f32` (`[0,1]`),
/// `f64` (scalar). For `u8`/`f32` the aarch64 path is NEON.
pub fn ycbcr_from_rgb<T>(src: &Image<T, 3>, dst: &mut Image<T, 3>) -> Result<(), ImageError>
where
    T: YuvFamily,
{
    T::ycc_from_rgb_impl(src, dst, ChromaOrder::YCrCb)
}

/// Convert a YCbCr image (`[Y, Cr, Cb]`, full range) back to RGB. Inverse of
/// [`ycbcr_from_rgb`].
pub fn rgb_from_ycbcr<T>(src: &Image<T, 3>, dst: &mut Image<T, 3>) -> Result<(), ImageError>
where
    T: YuvFamily,
{
    T::rgb_from_ycc_impl(src, dst, ChromaOrder::YCrCb)
}

/// Convert an RGB image to planar YUV, channel order `[Y, U=Cb, V=Cr]`, full range.
///
/// Same math as [`ycbcr_from_rgb`] with the two chroma channels swapped in storage.
pub fn yuv_from_rgb<T>(src: &Image<T, 3>, dst: &mut Image<T, 3>) -> Result<(), ImageError>
where
    T: YuvFamily,
{
    T::ycc_from_rgb_impl(src, dst, ChromaOrder::YuvCbCr)
}

/// Convert a planar YUV image (`[Y, U=Cb, V=Cr]`, full range) back to RGB. Inverse of
/// [`yuv_from_rgb`].
pub fn rgb_from_yuv<T>(src: &Image<T, 3>, dst: &mut Image<T, 3>) -> Result<(), ImageError>
where
    T: YuvFamily,
{
    T::rgb_from_ycc_impl(src, dst, ChromaOrder::YuvCbCr)
}

// ===== Family B: video decode (BT.601 limited range) ================================

#[inline]
fn check_dst_size(
    dst: &Image<u8, 3>,
    width: usize,
    height: usize,
    src_len: usize,
    expected_src: usize,
) -> Result<(), ImageError> {
    if dst.width() != width || dst.height() != height || src_len != expected_src {
        return Err(ImageError::InvalidImageSize(
            src_len,
            width,
            height,
            expected_src,
        ));
    }
    Ok(())
}

macro_rules! impl_packed422 {
    ($fn_name:ident, $variant:ident, $doc:expr_2021) => {
        #[doc = $doc]
        pub fn $fn_name(src: &[u8], dst: &mut Image<u8, 3>) -> Result<(), ImageError> {
            let (w, h) = (dst.width(), dst.height());
            check_dst_size(dst, w, h, src.len(), w * h * 2)?;
            kernels::rgb_from_packed422(src, dst.as_slice_mut(), w, h, Packed422::$variant);
            Ok(())
        }
    };
}

impl_packed422!(
    rgb_from_yuyv,
    Yuyv,
    "Decode a packed 4:2:2 YUYV (`Y0 U Y1 V`) buffer to RGB (BT.601 limited range)."
);
impl_packed422!(
    rgb_from_uyvy,
    Uyvy,
    "Decode a packed 4:2:2 UYVY (`U Y0 V Y1`) buffer to RGB (BT.601 limited range)."
);
impl_packed422!(
    rgb_from_yvyu,
    Yvyu,
    "Decode a packed 4:2:2 YVYU (`Y0 V Y1 U`) buffer to RGB (BT.601 limited range)."
);

/// Decode a planar 4:2:0 NV12 (Y plane + interleaved `UV`) buffer to RGB (BT.601 limited).
pub fn rgb_from_nv12(src: &[u8], dst: &mut Image<u8, 3>) -> Result<(), ImageError> {
    let (w, h) = (dst.width(), dst.height());
    check_dst_size(dst, w, h, src.len(), w * h * 3 / 2)?;
    let (y, uv) = src.split_at(w * h);
    kernels::rgb_from_planar420(y, uv, &[], dst.as_slice_mut(), w, h, Planar420::Nv12);
    Ok(())
}

/// Decode a planar 4:2:0 NV21 (Y plane + interleaved `VU`) buffer to RGB (BT.601 limited).
pub fn rgb_from_nv21(src: &[u8], dst: &mut Image<u8, 3>) -> Result<(), ImageError> {
    let (w, h) = (dst.width(), dst.height());
    check_dst_size(dst, w, h, src.len(), w * h * 3 / 2)?;
    let (y, uv) = src.split_at(w * h);
    kernels::rgb_from_planar420(y, uv, &[], dst.as_slice_mut(), w, h, Planar420::Nv21);
    Ok(())
}

/// Decode a planar 4:2:0 I420 (Y, then U plane, then V plane) buffer to RGB (BT.601 limited).
pub fn rgb_from_i420(src: &[u8], dst: &mut Image<u8, 3>) -> Result<(), ImageError> {
    let (w, h) = (dst.width(), dst.height());
    check_dst_size(dst, w, h, src.len(), w * h * 3 / 2)?;
    let n = w * h;
    let (y, chroma) = src.split_at(n);
    let (u, v) = chroma.split_at(n / 4);
    kernels::rgb_from_planar420(y, u, v, dst.as_slice_mut(), w, h, Planar420::I420);
    Ok(())
}

/// Decode a planar 4:2:0 YV12 (Y, then V plane, then U plane) buffer to RGB (BT.601 limited).
pub fn rgb_from_yv12(src: &[u8], dst: &mut Image<u8, 3>) -> Result<(), ImageError> {
    let (w, h) = (dst.width(), dst.height());
    check_dst_size(dst, w, h, src.len(), w * h * 3 / 2)?;
    let n = w * h;
    let (y, chroma) = src.split_at(n);
    let (v, u) = chroma.split_at(n / 4);
    // c0 = V plane, c1 = U plane (chroma_at handles the YV12 swap).
    kernels::rgb_from_planar420(y, v, u, dst.as_slice_mut(), w, h, Planar420::Yv12);
    Ok(())
}

/// Encode an RGB image to a packed 4:2:2 YUYV (`Y0 U Y1 V`) buffer (BT.601 limited).
///
/// `dst` must be exactly `width*height*2` bytes and `width` must be even. Luma is
/// per-pixel; the shared chroma of each horizontal pixel pair is their rounded
/// average. Inverse of [`rgb_from_yuyv`] (round-trips to within subsampling error).
pub fn yuyv_from_rgb(src: &Image<u8, 3>, dst: &mut [u8]) -> Result<(), ImageError> {
    let (w, h) = (src.width(), src.height());
    if w % 2 != 0 || dst.len() != w * h * 2 {
        return Err(ImageError::InvalidImageSize(dst.len(), w, h, w * h * 2));
    }
    kernels::yuyv_from_rgb(src.as_slice(), dst, w, h);
    Ok(())
}

/// Encode an RGB image to a planar 4:2:0 NV12 (Y plane + interleaved `UV`) buffer
/// (BT.601 limited).
///
/// `dst` must be exactly `width*height*3/2` bytes and both dimensions must be even.
/// Each chroma pair is the rounded average of its 2×2 luma block. Inverse of
/// [`rgb_from_nv12`].
pub fn nv12_from_rgb(src: &Image<u8, 3>, dst: &mut [u8]) -> Result<(), ImageError> {
    let (w, h) = (src.width(), src.height());
    if w % 2 != 0 || h % 2 != 0 || dst.len() != w * h * 3 / 2 {
        return Err(ImageError::InvalidImageSize(dst.len(), w, h, w * h * 3 / 2));
    }
    let (y, uv) = dst.split_at_mut(w * h);
    kernels::nv12_from_rgb(src.as_slice(), y, uv, w, h);
    Ok(())
}

/// The mode to convert YUV to RGB.
///
/// These modes correspond to ITU-R Broadcasting Television standards that define
/// the coefficients and ranges for YUV color space conversion:
///
/// ## Official ITU-R Documentation:
/// - **BT.601**: [ITU-R BT.601-7](https://www.itu.int/dms_pubrec/itu-r/rec/bt/R-REC-BT.601-7-201103-I!!PDF-E.pdf) - SDTV standard
/// - **BT.709**: [ITU-R BT.709-6](https://www.itu.int/rec/R-REC-BT.709-6-201506-I/en) - HDTV standard
/// - **BT.2020**: [ITU-R BT.2020-2](https://www.itu.int/rec/R-REC-BT.2020-2-201510-I/en) - Ultra HD standard
///
/// ## Additional Resources:
/// - [ITU-R BT.2407-0](https://www.itu.int/dms_pub/itu-r/opb/rep/R-REP-BT.2407-2017-PDF-E.pdf) - Color gamut conversion guide
/// - [Ultra HD Forum Guidelines](https://ultrahdforum.org/wp-content/uploads/UHD-Guidelines-V2.5-Fall2021.pdf) - Industry best practices
pub enum YuvToRgbMode {
    /// BT.601 full range (0-255 for Y, U, V).
    /// Used for SDTV, older cameras, and JPEG images.
    Bt601Full,
    /// BT.709 full range (0-255 for Y, U, V).
    /// Used for HDTV, modern displays, and sRGB content.
    Bt709Full,
    /// BT.601 limited range (16-235 for Y, 16-240 for U, V).
    /// Used for broadcast television and professional video equipment.
    Bt601Limited,
}

/// Convert a YUYV image to an RGB image.
///
/// # Arguments
///
/// * `src` - The YUYV image data.
/// * `dst` - The RGB image to store the result.
/// * `mode` - The mode to convert YUV to RGB.
///
/// # Returns
///
/// The RGB image in HxWx3 format.
pub fn convert_yuyv_to_rgb_u8(
    src: &[u8],
    dst: &mut Image<u8, 3>,
    mode: YuvToRgbMode,
) -> Result<(), ImageError> {
    // the yuyv image is 2 bytes per pixel, so we need to divide by 2
    let (width, height) = (dst.width(), dst.height());
    if src.len() != width * height * 2 {
        return Err(ImageError::InvalidImageSize(
            src.len(),
            width,
            height,
            width * height * 2,
        ));
    }

    let rgb_data = dst.as_slice_mut();
    let rgb_row_len = width * 3;
    let yuyv_row_len = width * 2;

    const ROWS_PER_TASK: usize = 16;
    rgb_data
        .par_chunks_mut(ROWS_PER_TASK * rgb_row_len)
        .enumerate()
        .for_each(|(chunk_idx, rgb_chunk)| {
            let row_base = chunk_idx * ROWS_PER_TASK;
            rgb_chunk
                .chunks_exact_mut(rgb_row_len)
                .enumerate()
                .for_each(|(dr, rgb_row)| {
                    let row = row_base + dr;
                    let yuyv_row_start = row * yuyv_row_len;
                    let yuyv_row = &src[yuyv_row_start..yuyv_row_start + yuyv_row_len];

                    rgb_row
                        .chunks_exact_mut(6)
                        .enumerate()
                        .for_each(|(col, rgb_chunk)| {
                            let yuyv_idx = col * 4;
                            if yuyv_idx + 3 < yuyv_row.len() {
                                let y0 = yuyv_row[yuyv_idx];
                                let u = yuyv_row[yuyv_idx + 1];
                                let y1 = yuyv_row[yuyv_idx + 2];
                                let v = yuyv_row[yuyv_idx + 3];

                                let (r0, g0, b0) = match mode {
                                    YuvToRgbMode::Bt601Full => yuv_to_rgb_u8_bt601_full(y0, u, v),
                                    YuvToRgbMode::Bt709Full => yuv_to_rgb_u8_bt709_full(y0, u, v),
                                    YuvToRgbMode::Bt601Limited => {
                                        yuv_to_rgb_u8_bt601_limited(y0, u, v)
                                    }
                                };
                                let (r1, g1, b1) = match mode {
                                    YuvToRgbMode::Bt601Full => yuv_to_rgb_u8_bt601_full(y1, u, v),
                                    YuvToRgbMode::Bt709Full => yuv_to_rgb_u8_bt709_full(y1, u, v),
                                    YuvToRgbMode::Bt601Limited => {
                                        yuv_to_rgb_u8_bt601_limited(y1, u, v)
                                    }
                                };

                                rgb_chunk[0] = r0;
                                rgb_chunk[1] = g0;
                                rgb_chunk[2] = b0;
                                rgb_chunk[3] = r1;
                                rgb_chunk[4] = g1;
                                rgb_chunk[5] = b1;
                            }
                        });
                });
        });

    Ok(())
}

#[inline]
fn yuv_to_rgb_u8_bt601_full(y: u8, u: u8, v: u8) -> (u8, u8, u8) {
    // Convert to signed integers and apply offsets
    let y_val = y as i32;
    let u_val = (u as i32) - 128;
    let v_val = (v as i32) - 128;

    // Fixed-point coefficients (multiplied by 1024)
    // 1.402 * 1024 = 1436
    // 0.344136 * 1024 = 352
    // 0.714136 * 1024 = 731
    // 1.772 * 1024 = 1815

    let r = y_val + ((1436 * v_val + 512) >> 10);
    let g = y_val - ((352 * u_val + 731 * v_val + 512) >> 10);
    let b = y_val + ((1815 * u_val + 512) >> 10);

    (
        r.clamp(0, 255) as u8,
        g.clamp(0, 255) as u8,
        b.clamp(0, 255) as u8,
    )
}

#[inline]
fn yuv_to_rgb_u8_bt709_full(y: u8, u: u8, v: u8) -> (u8, u8, u8) {
    let y_val = y as i32;
    let u_val = (u as i32) - 128;
    let v_val = (v as i32) - 128;

    // BT.709 coefficients * 1024
    // 1.5748 * 1024 = 1612
    // 0.187324 * 1024 = 192
    // 0.468124 * 1024 = 479
    // 1.8556 * 1024 = 1900

    let r = y_val + ((1612 * v_val + 512) >> 10);
    let g = y_val - ((192 * u_val + 479 * v_val + 512) >> 10);
    let b = y_val + ((1900 * u_val + 512) >> 10);

    (
        r.clamp(0, 255) as u8,
        g.clamp(0, 255) as u8,
        b.clamp(0, 255) as u8,
    )
}

#[inline]
fn yuv_to_rgb_u8_bt601_limited(y: u8, u: u8, v: u8) -> (u8, u8, u8) {
    // Apply range scaling first
    let y_val = ((y as i32 - 16) * 1192 + 512) >> 10; // 1192 = 1.164 * 1024
    let u_val = (u as i32) - 128;
    let v_val = (v as i32) - 128;

    // BT.601 limited coefficients * 1024
    let r = y_val + ((1634 * v_val + 512) >> 10);
    let g = y_val - ((401 * u_val + 832 * v_val + 512) >> 10);
    let b = y_val + ((2066 * u_val + 512) >> 10);

    (
        r.clamp(0, 255) as u8,
        g.clamp(0, 255) as u8,
        b.clamp(0, 255) as u8,
    )
}
