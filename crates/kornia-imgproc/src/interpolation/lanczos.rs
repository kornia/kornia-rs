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
    let z = std::f32::consts::PI * r;
    let z2 = z * z;
    let mut p = -2.505_210_8e-8_f32;
    p = p * z2 + 2.755_731_9e-6;
    p = p * z2 + -1.984_127e-4;
    p = p * z2 + 8.333_334e-3;
    p = p * z2 + -1.666_666_7e-1;
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
    const PI: f32 = std::f32::consts::PI;
    if x.abs() < 1e-5 {
        return 1.0;
    }
    if x.abs() >= 3.0 {
        return 0.0;
    }
    let pix = PI * x;
    let pix3 = pix * 0.333_333_34;
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

/// Direct 6×6 Lanczos-3 sample at coordinate `(sx, sy)` for channel `c` —
/// the warp kernels' inner loop: per-axis weights normalized separately, then
/// a two-level fused accumulation (per-row `fmaf` over `dx`, then `fmaf` of
/// the row result by `wy[dy]`).
#[inline(never)]
pub(crate) fn lanczos_sample<const C: usize>(
    image: &kornia_image::Image<f32, C>,
    sx: f32,
    sy: f32,
    c: usize,
) -> f32 {
    let (rows, cols) = (image.rows() as i64, image.cols() as i64);
    let s = image.as_slice();

    let x0 = sx.floor();
    let y0 = sy.floor();
    let frac_x = sx - x0;
    let frac_y = sy - y0;
    let (x0, y0) = (x0 as i64, y0 as i64);

    let mut wx = [
        lanczos3(frac_x + 2.0),
        lanczos3(frac_x + 1.0),
        lanczos3(frac_x),
        lanczos3(frac_x - 1.0),
        lanczos3(frac_x - 2.0),
        lanczos3(frac_x - 3.0),
    ];
    let mut wy = [
        lanczos3(frac_y + 2.0),
        lanczos3(frac_y + 1.0),
        lanczos3(frac_y),
        lanczos3(frac_y - 1.0),
        lanczos3(frac_y - 2.0),
        lanczos3(frac_y - 3.0),
    ];
    let sum_wx = wx[0] + wx[1] + wx[2] + wx[3] + wx[4] + wx[5];
    let sum_wy = wy[0] + wy[1] + wy[2] + wy[3] + wy[4] + wy[5];
    let inv_x = 1.0 / sum_wx;
    let inv_y = 1.0 / sum_wy;
    for w in &mut wx {
        *w *= inv_x;
    }
    for w in &mut wy {
        *w *= inv_y;
    }

    let mut acc = 0.0f32;
    for (dy, &wyv) in wy.iter().enumerate() {
        let yi = (y0 + dy as i64 - 2).clamp(0, rows - 1) as usize;
        let row = yi * cols as usize * C;
        let mut rx = 0.0f32;
        for (dx, &wxv) in wx.iter().enumerate() {
            let xi = (x0 + dx as i64 - 2).clamp(0, cols - 1) as usize;
            rx = wxv.mul_add(s[row + xi * C + c], rx);
        }
        acc = wyv.mul_add(rx, acc);
    }
    acc
}

/// Separable Lanczos-3 resize on the half-pixel grid — the byte-exact CPU
/// twin of the CUDA `resize_lanczos_{h,v}_3c` pipeline: identical tables from
/// [`lanczos_axis`], identical H-then-V pass structure with an f32
/// intermediate of `dst_w × src_h`, identical fused accumulation.
pub(crate) fn resize_lanczos_separable<const C: usize>(
    src: &kornia_image::Image<f32, C>,
    dst: &mut kornia_image::Image<f32, C>,
) {
    let (src_w, src_h) = (src.cols(), src.rows());
    let (dst_w, dst_h) = (dst.cols(), dst.rows());
    let (x0s, wx) = lanczos_axis(src_w, dst_w);
    let (y0s, wy) = lanczos_axis(src_h, dst_h);
    let s = src.as_slice();

    // Pass 1 — horizontal: (src_w, src_h) → (dst_w, src_h).
    let mut inter = vec![0.0f32; dst_w * src_h * C];
    for sy in 0..src_h {
        let srow = &s[sy * src_w * C..(sy + 1) * src_w * C];
        let irow = &mut inter[sy * dst_w * C..(sy + 1) * dst_w * C];
        for dx in 0..dst_w {
            let x0 = x0s[dx] as i64;
            let w = &wx[dx * 6..dx * 6 + 6];
            for k in 0..C {
                let mut acc = 0.0f32;
                for (t, &wt) in w.iter().enumerate() {
                    let xi = (x0 + t as i64 - 2).clamp(0, src_w as i64 - 1) as usize;
                    acc = wt.mul_add(srow[xi * C + k], acc);
                }
                irow[dx * C + k] = acc;
            }
        }
    }

    // Pass 2 — vertical: (dst_w, src_h) → (dst_w, dst_h).
    let d = dst.as_slice_mut();
    for dy in 0..dst_h {
        let y0 = y0s[dy] as i64;
        let w = &wy[dy * 6..dy * 6 + 6];
        let drow = &mut d[dy * dst_w * C..(dy + 1) * dst_w * C];
        for dx in 0..dst_w {
            for k in 0..C {
                let mut acc = 0.0f32;
                for (t, &wt) in w.iter().enumerate() {
                    let yi = (y0 + t as i64 - 2).clamp(0, src_h as i64 - 1) as usize;
                    acc = wt.mul_add(inter[(yi * dst_w + dx) * C + k], acc);
                }
                drow[dx * C + k] = acc;
            }
        }
    }
}
