//! Sepia tone (Kornia parity).
//!
//! Applies the standard sepia color matrix to an RGB image, clamping to range:
//! ```text
//! R' = 0.393 R + 0.769 G + 0.189 B
//! G' = 0.349 R + 0.686 G + 0.168 B
//! B' = 0.272 R + 0.534 G + 0.131 B
//! ```
//! The f32 path reuses the shared [`matrix3_affine_f32`] kernel; the u8 path uses
//! a NEON widening multiply-accumulate in Q8 fixed point with a scalar oracle.

use kornia_image::{Image, ImageError};

use super::kernel_common::check_size;
use super::matrix::matrix3_affine_f32;

/// Sepia matrix, row-major (R',G',B' rows).
const SEPIA_M: [f32; 9] = [
    0.393, 0.769, 0.189, // R'
    0.349, 0.686, 0.168, // G'
    0.272, 0.534, 0.131, // B'
];

/// Apply sepia tone to an RGB f32 image (values in any range; matrix is linear).
///
/// # Errors
/// Returns [`ImageError::InvalidImageSize`] if `src` and `dst` differ in size.
pub fn sepia_from_rgb_f32(src: &Image<f32, 3>, dst: &mut Image<f32, 3>) -> Result<(), ImageError> {
    check_size(src, dst)?;
    #[cfg(feature = "cuda")]
    {
        use super::cuda_dispatch::{pair_residency, Residency};
        if let Residency::Device(exec) = pair_residency(src, dst)? {
            return exec
                .run(|stream| super::cuda_dispatch::sepia_from_rgb_f32_cuda(src, dst, stream));
        }
    }
    matrix3_affine_f32(
        src.as_slice(),
        dst.as_slice_mut(),
        src.rows() * src.cols(),
        SEPIA_M,
        [0.0; 3],
    );
    Ok(())
}

/// Apply sepia tone to an RGB u8 image, saturating to `[0, 255]`.
///
/// # Errors
/// Returns [`ImageError::InvalidImageSize`] if `src` and `dst` differ in size.
pub fn sepia_from_rgb_u8(src: &Image<u8, 3>, dst: &mut Image<u8, 3>) -> Result<(), ImageError> {
    check_size(src, dst)?;
    #[cfg(feature = "cuda")]
    {
        use super::cuda_dispatch::{pair_residency, Residency};
        if let Residency::Device(exec) = pair_residency(src, dst)? {
            return exec
                .run(|stream| super::cuda_dispatch::sepia_from_rgb_u8_cuda(src, dst, stream));
        }
    }
    let n = src.rows() * src.cols();
    super::kernel_common::par_strip_dispatch(
        src.as_slice(),
        dst.as_slice_mut(),
        n,
        3,
        16,
        sepia_u8_kernel,
    );
    Ok(())
}

#[inline]
fn sepia_u8_kernel(src: &[u8], dst: &mut [u8], npixels: usize) {
    #[cfg(target_arch = "aarch64")]
    {
        sepia_u8_neon(src, dst, npixels);
        return;
    }
    #[allow(unreachable_code)]
    sepia_u8_scalar(src, dst, npixels);
}

/// Q8 fixed-point sepia coefficients: `round(coeff * 256)`.
const Q: [u16; 9] = [
    101, 197, 48, // R': .393,.769,.189
    89, 176, 43, // G': .349,.686,.168
    70, 137, 34, // B': .272,.534,.131
];

/// Scalar oracle: Q8 fixed-point MAC, rounded, saturated. Matches NEON exactly.
pub(crate) fn sepia_u8_scalar(src: &[u8], dst: &mut [u8], npixels: usize) {
    for i in 0..npixels {
        let si = i * 3;
        let (r, g, b) = (src[si] as u32, src[si + 1] as u32, src[si + 2] as u32);
        let mac = |a: u16, c: u16, d: u16| -> u8 {
            let v = (a as u32 * r + c as u32 * g + d as u32 * b + 128) >> 8;
            v.min(255) as u8
        };
        dst[si] = mac(Q[0], Q[1], Q[2]);
        dst[si + 1] = mac(Q[3], Q[4], Q[5]);
        dst[si + 2] = mac(Q[6], Q[7], Q[8]);
    }
}

/// NEON u8 sepia: 16 px/iter via `vld3q_u8`, widening Q8 MAC, `vst3q_u8`.
///
/// Each output channel is `(Σ coeff·chan + 128) >> 8` saturated to u8. The
/// 8-bit inputs and Q8 coeffs keep partial products in u16, but the three-term
/// sum can exceed u16, so we accumulate in u32 (split low/high 8-lane halves).
#[cfg(target_arch = "aarch64")]
fn sepia_u8_neon(src: &[u8], dst: &mut [u8], npixels: usize) {
    use std::arch::aarch64::*;
    unsafe {
        let sp = src.as_ptr();
        let dp = dst.as_mut_ptr();

        // One output channel for an 8-lane u8 group: widen each input lane to
        // u16, MAC with Q8 coeffs into u32 (two 4-lane halves), round + shift,
        // narrow back. Returns a u16x8 of results in [0, ~ up to 255+].
        #[inline(always)]
        unsafe fn chan8(
            r: uint16x8_t,
            g: uint16x8_t,
            b: uint16x8_t,
            cr: u16,
            cg: u16,
            cb: u16,
        ) -> uint16x8_t {
            unsafe {
                // u32 accumulate to avoid overflow (max ~ 255*(101+197+48)=88230 > u16).
                let mut lo = vmull_n_u16(vget_low_u16(r), cr);
                lo = vmlal_n_u16(lo, vget_low_u16(g), cg);
                lo = vmlal_n_u16(lo, vget_low_u16(b), cb);
                let mut hi = vmull_n_u16(vget_high_u16(r), cr);
                hi = vmlal_n_u16(hi, vget_high_u16(g), cg);
                hi = vmlal_n_u16(hi, vget_high_u16(b), cb);
                // (acc + 128) >> 8, rounding narrowing shift to u16.
                vcombine_u16(vrshrn_n_u32::<8>(lo), vrshrn_n_u32::<8>(hi))
            }
        }

        let bulk = npixels & !15;
        let mut i = 0usize;
        while i < bulk {
            let px = vld3q_u8(sp.add(i * 3));
            let r_lo = vmovl_u8(vget_low_u8(px.0));
            let r_hi = vmovl_u8(vget_high_u8(px.0));
            let g_lo = vmovl_u8(vget_low_u8(px.1));
            let g_hi = vmovl_u8(vget_high_u8(px.1));
            let b_lo = vmovl_u8(vget_low_u8(px.2));
            let b_hi = vmovl_u8(vget_high_u8(px.2));

            let or_lo = chan8(r_lo, g_lo, b_lo, Q[0], Q[1], Q[2]);
            let or_hi = chan8(r_hi, g_hi, b_hi, Q[0], Q[1], Q[2]);
            let og_lo = chan8(r_lo, g_lo, b_lo, Q[3], Q[4], Q[5]);
            let og_hi = chan8(r_hi, g_hi, b_hi, Q[3], Q[4], Q[5]);
            let ob_lo = chan8(r_lo, g_lo, b_lo, Q[6], Q[7], Q[8]);
            let ob_hi = chan8(r_hi, g_hi, b_hi, Q[6], Q[7], Q[8]);

            // Saturating narrow u16 -> u8 (clamps the >255 overflow to 255).
            let out = uint8x16x3_t(
                vcombine_u8(vqmovn_u16(or_lo), vqmovn_u16(or_hi)),
                vcombine_u8(vqmovn_u16(og_lo), vqmovn_u16(og_hi)),
                vcombine_u8(vqmovn_u16(ob_lo), vqmovn_u16(ob_hi)),
            );
            vst3q_u8(dp.add(i * 3), out);
            i += 16;
        }
        // Scalar tail.
        if i < npixels {
            sepia_u8_scalar(&src[i * 3..], &mut dst[i * 3..], npixels - i);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use kornia_image::ImageSize;

    #[test]
    fn sepia_u8_known_value() {
        // Pure white -> sepia of (255,255,255).
        // R' = round(255*(101+197+48)/256) = round(255*346/256)=round(344.6)=345 -> 255
        // G' = 255*(89+176+43)/256 = 255*308/256 = 306.8 -> 255
        // B' = 255*(70+137+34)/256 = 255*241/256 = 240.04 -> 240
        let src = Image::<u8, 3>::from_size_val(
            ImageSize {
                width: 1,
                height: 1,
            },
            255,
        )
        .unwrap();
        let mut dst = Image::<u8, 3>::from_size_val(src.size(), 0).unwrap();
        sepia_from_rgb_u8(&src, &mut dst).unwrap();
        assert_eq!(dst.as_slice(), &[255, 255, 240]);
    }

    #[test]
    fn sepia_u8_neon_matches_scalar() {
        // 37 px (16-wide block + tail) of varied values.
        let n = 37;
        let data: Vec<u8> = (0..n * 3).map(|i| ((i * 53 + 7) % 256) as u8).collect();
        let src = Image::<u8, 3>::new(
            ImageSize {
                width: n,
                height: 1,
            },
            data.clone(),
        )
        .unwrap();
        let mut got = Image::<u8, 3>::from_size_val(src.size(), 0).unwrap();
        sepia_from_rgb_u8(&src, &mut got).unwrap();
        let mut oracle = vec![0u8; n * 3];
        sepia_u8_scalar(&data, &mut oracle, n);
        assert_eq!(got.as_slice(), oracle.as_slice());
    }

    #[test]
    fn sepia_f32_known_value() {
        let src = Image::<f32, 3>::new(
            ImageSize {
                width: 1,
                height: 1,
            },
            vec![100.0, 150.0, 200.0],
        )
        .unwrap();
        let mut dst = Image::<f32, 3>::from_size_val(src.size(), 0.0).unwrap();
        sepia_from_rgb_f32(&src, &mut dst).unwrap();
        let exp_r = 0.393 * 100.0 + 0.769 * 150.0 + 0.189 * 200.0;
        let exp_g = 0.349 * 100.0 + 0.686 * 150.0 + 0.168 * 200.0;
        let exp_b = 0.272 * 100.0 + 0.534 * 150.0 + 0.131 * 200.0;
        let d = dst.as_slice();
        assert!((d[0] - exp_r).abs() < 1e-3);
        assert!((d[1] - exp_g).abs() < 1e-3);
        assert!((d[2] - exp_b).abs() < 1e-3);
    }
}
