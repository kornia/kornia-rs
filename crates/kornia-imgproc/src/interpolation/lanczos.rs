//! Lanczos-3 interpolation — weight math shared byte-for-byte with the CUDA
//! kernels.
//!
//! Two shapes exist, mirroring the kernels exactly:
//! * **resize** is separable: per-axis tap bases and normalized 6-weight
//!   tables built once by [`lanczos_axis`] — the same call feeds both the CPU
//!   passes and the CUDA launcher (which uploads the tables), so the two
//!   backends agree bit-for-bit by construction.
//! * **warps** are direct 6×6 with per-pixel weights: [`lanczos3`] here is the
//!   textual twin of the kernels' `lanczos3`/`sin_pi` device functions —
//!   plain mul/add polynomial (no libm `sin`, whose rounding differs between
//!   host libm and the CUDA math library), evaluated identically under
//!   `--fmad=false`.

/// `sin(π·x)` via exact integer reduction and an odd Taylor polynomial in
/// plain mul/add — the Rust twin of the kernels' `sin_pi`. Keep the
/// expression shapes in sync with the CUDA sources or byte-exactness dies.
#[inline]
pub(crate) fn sin_pi(x: f32) -> f32 {
    let k = x.round();
    let r = x - k;
    let z = 3.141_592_653_589_79_f32 * r;
    let z2 = z * z;
    let mut p = -2.505_210_838_5e-8_f32;
    p = p * z2 + 2.755_731_922_4e-6;
    p = p * z2 + -1.984_126_984_1e-4;
    p = p * z2 + 8.333_333_333_3e-3;
    p = p * z2 + -1.666_666_666_7e-1;
    let s = z + z * z2 * p;
    if (k as i32) & 1 != 0 {
        -s
    } else {
        s
    }
}

/// 3-lobe Lanczos window — twin of the kernels' `lanczos3`.
#[inline]
pub(crate) fn lanczos3(x: f32) -> f32 {
    const PI: f32 = 3.141_592_653_589_79;
    if x.abs() < 1e-5 {
        return 1.0;
    }
    if x.abs() >= 3.0 {
        return 0.0;
    }
    let pix = PI * x;
    let pix3 = pix * 0.333_333_33;
    sin_pi(x) * sin_pi(x * (1.0 / 3.0)) / (pix * pix3)
}

/// Per-axis separable-resize tables: for each destination index, the tap base
/// `x0` and the six normalized weights, on the half-pixel grid with the same
/// coordinate expression as the bilinear/nearest byte-exact contract
/// (`a*x + b`, plain ops, clamped to `[0, src-1]`).
///
/// Feeds both the CPU separable passes and the CUDA launcher (which uploads
/// the tables verbatim), so CPU and GPU resize-lanczos share every weight bit.
pub(crate) fn lanczos_axis(src_len: usize, dst_len: usize) -> (Vec<i32>, Vec<f32>) {
    let a = src_len as f32 / dst_len as f32;
    let b = 0.5 * a - 0.5;
    let max = (src_len - 1) as f32;

    let mut x0s = Vec::with_capacity(dst_len);
    let mut weights = Vec::with_capacity(dst_len * 6);
    for i in 0..dst_len {
        let s = (a * i as f32 + b).clamp(0.0, max);
        let x0 = s.floor();
        let frac = s - x0;
        x0s.push(x0 as i32);

        let mut w = [
            lanczos3(frac + 2.0),
            lanczos3(frac + 1.0),
            lanczos3(frac),
            lanczos3(frac - 1.0),
            lanczos3(frac - 2.0),
            lanczos3(frac - 3.0),
        ];
        // Same normalization expression order as the pre-table kernel: a plain
        // left-to-right sum, one reciprocal, six multiplies.
        let sum = w[0] + w[1] + w[2] + w[3] + w[4] + w[5];
        let inv = 1.0 / sum;
        for wi in &mut w {
            *wi *= inv;
        }
        weights.extend_from_slice(&w);
    }
    (x0s, weights)
}
