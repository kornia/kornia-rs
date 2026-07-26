//! Bicubic (Keys, a = −0.5) interpolation — the byte-exact CPU twin of the
//! CUDA bicubic kernels.
//!
//! Weight polynomials use `mul_add` where the kernels say `fmaf`, and plain
//! mul/add where they say so; the 4×4 accumulation walks the taps in the
//! kernel's loop order (`dy` outer, `dx` inner, one fused accumulator per
//! channel). Keep every expression shape in sync with
//! `warp_affine_bicubic_3c` / `resize_bicubic_3c` or byte-exactness dies.

use kornia_image::Image;

/// Horner-form Keys cubic weights for `frac ∈ [0, 1)` — twin of the kernels'
/// weight block (explicit fmaf chain).
#[inline]
pub(crate) fn keys_weights(frac: f32) -> [f32; 4] {
    let mut w = [0.0f32; 4];
    let mut t;
    t = 1.0 + frac;
    w[0] = (-0.5f32).mul_add(t, 2.5).mul_add(t, -4.0).mul_add(t, 2.0);
    t = frac;
    w[1] = (1.5f32.mul_add(t, -2.5) * t).mul_add(t, 1.0);
    t = 1.0 - frac;
    w[2] = (1.5f32.mul_add(t, -2.5) * t).mul_add(t, 1.0);
    t = 2.0 - frac;
    w[3] = (-0.5f32).mul_add(t, 2.5).mul_add(t, -4.0).mul_add(t, 2.0);
    w
}

/// Direct 4×4 bicubic sample at (already validated / clamped) coordinate
/// `(sx, sy)`, taps replicate-clamped per axis — the kernels' inner loop,
/// channel `c` only.
#[inline(never)]
pub(crate) fn bicubic_sample<const C: usize>(
    image: &Image<f32, C>,
    sx: f32,
    sy: f32,
    c: usize,
) -> f32 {
    let (rows, cols) = (image.rows() as i64, image.cols() as i64);
    let s = image.as_slice();

    let x0 = sx.floor();
    let y0 = sy.floor();
    let wx = keys_weights(sx - x0);
    let wy = keys_weights(sy - y0);
    let (x0, y0) = (x0 as i64, y0 as i64);

    let mut acc = 0.0f32;
    for (dy, &wyv) in wy.iter().enumerate() {
        let yi = (y0 + dy as i64 - 1).clamp(0, rows - 1) as usize;
        let row = yi * cols as usize * C;
        for (dx, &wxv) in wx.iter().enumerate() {
            let xi = (x0 + dx as i64 - 1).clamp(0, cols - 1) as usize;
            // Kernel order: w = wx[dx] * wy[dy] (plain mul), then
            // acc = fmaf(w, v, acc).
            let w = wxv * wyv;
            acc = w.mul_add(s[row + xi * C + c], acc);
        }
    }
    acc
}
