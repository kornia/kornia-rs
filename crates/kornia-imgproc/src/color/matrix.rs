//! Generic 3×3 affine color-matrix kernel: `out = M · [c0,c1,c2]ᵀ + bias`.
//!
//! This is the shared centerpiece for every linear color transform — RGB↔XYZ,
//! RGB↔YCbCr, RGB↔YUV, and the linear stage of Lab/Luv. The matrix `M` (row-major,
//! 9 floats) and `bias` (3 floats) are passed in so one kernel serves all of them.
//!
//! Layout per pixel: input is 3 interleaved f32 (`vld3q_f32` deinterleaves to lanes),
//! output is 3 interleaved f32 (`vst3q_f32`). Bulk 8 px/iter (2× structured load),
//! 4-px remainder, scalar tail — mirrors `gray_from_rgb_f32_neon`.

use super::kernel_common::par_strip_dispatch;

/// Apply a 3×3 affine transform `out = M·in + bias` to an interleaved 3-channel f32 image.
///
/// `m` is row-major: `out[0] = m[0]*in0 + m[1]*in1 + m[2]*in2 + bias[0]`, etc.
pub(crate) fn matrix3_affine_f32(
    src: &[f32],
    dst: &mut [f32],
    npixels: usize,
    m: [f32; 9],
    bias: [f32; 3],
) {
    debug_assert!(src.len() >= npixels * 3);
    debug_assert!(dst.len() >= npixels * 3);
    par_strip_dispatch(src, dst, npixels, 3, 8, move |s, d, n| {
        matrix3_affine_f32_kernel(s, d, n, &m, &bias)
    });
}

#[inline]
fn matrix3_affine_f32_kernel(src: &[f32], dst: &mut [f32], npixels: usize, m: &[f32; 9], b: &[f32; 3]) {
    #[cfg(target_arch = "aarch64")]
    {
        matrix3_affine_f32_neon(src, dst, npixels, m, b);
        return;
    }
    #[cfg(target_arch = "x86_64")]
    {
        let cpu = crate::simd::cpu_features();
        if cpu.has_avx2 && cpu.has_fma {
            // SAFETY: AVX2+FMA confirmed by the runtime probe.
            unsafe { matrix3_affine_f32_avx2(src, dst, npixels, m, b) };
            return;
        }
    }
    #[allow(unreachable_code)]
    matrix3_affine_f32_scalar(src, dst, npixels, m, b);
}

/// NEON: 8 px/iter via 2× `vld3q_f32`, 9 `vfmaq`-style MACs + bias, `vst3q_f32`.
#[cfg(target_arch = "aarch64")]
fn matrix3_affine_f32_neon(src: &[f32], dst: &mut [f32], npixels: usize, m: &[f32; 9], b: &[f32; 3]) {
    use std::arch::aarch64::*;
    unsafe {
        let (m00, m01, m02) = (vdupq_n_f32(m[0]), vdupq_n_f32(m[1]), vdupq_n_f32(m[2]));
        let (m10, m11, m12) = (vdupq_n_f32(m[3]), vdupq_n_f32(m[4]), vdupq_n_f32(m[5]));
        let (m20, m21, m22) = (vdupq_n_f32(m[6]), vdupq_n_f32(m[7]), vdupq_n_f32(m[8]));
        let (b0, b1, b2) = (vdupq_n_f32(b[0]), vdupq_n_f32(b[1]), vdupq_n_f32(b[2]));
        let sp = src.as_ptr();
        let dp = dst.as_mut_ptr();

        let transform = |p: float32x4x3_t| -> float32x4x3_t {
            let o0 = vfmaq_f32(vfmaq_f32(vfmaq_f32(b0, p.0, m00), p.1, m01), p.2, m02);
            let o1 = vfmaq_f32(vfmaq_f32(vfmaq_f32(b1, p.0, m10), p.1, m11), p.2, m12);
            let o2 = vfmaq_f32(vfmaq_f32(vfmaq_f32(b2, p.0, m20), p.1, m21), p.2, m22);
            float32x4x3_t(o0, o1, o2)
        };

        let bulk8 = npixels & !7;
        let mut i = 0usize;
        while i < bulk8 {
            let a = vld3q_f32(sp.add(i * 3));
            let c = vld3q_f32(sp.add((i + 4) * 3));
            vst3q_f32(dp.add(i * 3), transform(a));
            vst3q_f32(dp.add((i + 4) * 3), transform(c));
            i += 8;
        }
        if i + 4 <= npixels {
            let a = vld3q_f32(sp.add(i * 3));
            vst3q_f32(dp.add(i * 3), transform(a));
            i += 4;
        }
        while i < npixels {
            let si = i * 3;
            let (c0, c1, c2) = (*sp.add(si), *sp.add(si + 1), *sp.add(si + 2));
            *dp.add(si) = b[0] + m[0] * c0 + m[1] * c1 + m[2] * c2;
            *dp.add(si + 1) = b[1] + m[3] * c0 + m[4] * c1 + m[5] * c2;
            *dp.add(si + 2) = b[2] + m[6] * c0 + m[7] * c1 + m[8] * c2;
            i += 1;
        }
    }
}

/// AVX2+FMA: 8 px/iter. Reuses the gray f32 deinterleave/interleave permute layout.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn matrix3_affine_f32_avx2(
    src: &[f32],
    dst: &mut [f32],
    npixels: usize,
    m: &[f32; 9],
    b: &[f32; 3],
) {
    // Scalar fallback body for x86 correctness; the AVX2 deinterleave shuffle is
    // identical to gray's and will be specialized during the x86 perf pass.
    matrix3_affine_f32_scalar(src, dst, npixels, m, b);
}

/// Portable scalar reference / oracle.
fn matrix3_affine_f32_scalar(src: &[f32], dst: &mut [f32], npixels: usize, m: &[f32; 9], b: &[f32; 3]) {
    for i in 0..npixels {
        let si = i * 3;
        let (c0, c1, c2) = (src[si], src[si + 1], src[si + 2]);
        dst[si] = b[0] + m[0] * c0 + m[1] * c1 + m[2] * c2;
        dst[si + 1] = b[1] + m[3] * c0 + m[4] * c1 + m[5] * c2;
        dst[si + 2] = b[2] + m[6] * c0 + m[7] * c1 + m[8] * c2;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn identity_matrix_is_noop() {
        let src: Vec<f32> = (0..30).map(|v| v as f32).collect();
        let mut dst = vec![0.0f32; 30];
        let id = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0];
        matrix3_affine_f32(&src, &mut dst, 10, id, [0.0; 3]);
        assert_eq!(src, dst);
    }

    #[test]
    fn affine_matches_scalar_odd_width() {
        // 7 px exercises the 4-px remainder + scalar tail of the NEON path.
        let src: Vec<f32> = (0..21).map(|v| v as f32 / 21.0).collect();
        let m = [0.4, 0.3, 0.2, 0.1, 0.5, 0.4, 0.2, 0.2, 0.6];
        let bias = [0.01, 0.02, 0.03];
        let mut simd = vec![0.0f32; 21];
        let mut scalar = vec![0.0f32; 21];
        matrix3_affine_f32(&src, &mut simd, 7, m, bias);
        matrix3_affine_f32_scalar(&src, &mut scalar, 7, &m, &bias);
        for (a, b) in simd.iter().zip(scalar.iter()) {
            assert!((a - b).abs() < 1e-6, "{a} != {b}");
        }
    }
}
