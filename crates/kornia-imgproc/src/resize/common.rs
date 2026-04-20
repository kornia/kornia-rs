//! Filter kinds and coefficient LUTs shared by bicubic/lanczos resize.
//!
//! The Q14 separable pipeline in [`super::separable`] consumes the output of
//! [`precompute_contribs`] (per-row `(offset, weights, ksize)`) plus the packed
//! `(xsrc, xw)` LUT produced by [`build_xsrc_lut`] + [`pack_xw_i16`].

/// Which separable kernel to use. `Cubic` is bicubic (a = -0.5), `Lanczos3` is
/// 3-lobe Lanczos.
#[derive(Copy, Clone)]
pub(super) enum FilterKind {
    Cubic,
    Lanczos3,
}

impl FilterKind {
    pub(super) fn support(self) -> f64 {
        match self {
            FilterKind::Cubic => 2.0,
            FilterKind::Lanczos3 => 3.0,
        }
    }

    pub(super) fn weight(self, x: f64) -> f64 {
        let ax = x.abs();
        match self {
            FilterKind::Cubic => {
                let a = -0.5;
                if ax < 1.0 {
                    (a + 2.0) * ax * ax * ax - (a + 3.0) * ax * ax + 1.0
                } else if ax < 2.0 {
                    a * ax * ax * ax - 5.0 * a * ax * ax + 8.0 * a * ax - 4.0 * a
                } else {
                    0.0
                }
            }
            FilterKind::Lanczos3 => {
                if ax < 1e-12 {
                    1.0
                } else if ax < 3.0 {
                    let px = std::f64::consts::PI * x;
                    let s = px.sin();
                    let s3 = (px / 3.0).sin();
                    3.0 * s * s3 / (px * px)
                } else {
                    0.0
                }
            }
        }
    }
}

/// Precompute per-output-pixel `(offset, weights, ksize)`.
///
/// When `antialias` is true (PIL / torchvision `antialias=True` semantics),
/// the filter is widened by the downscale factor so the kernel pre-filters
/// aliasing. This matches PIL's output within Q14 rounding noise but grows
/// `ksize` linearly with scale — e.g. bicubic 1080→224 uses ksize≈20.
///
/// When `antialias` is false (OpenCV `INTER_CUBIC` / `INTER_LANCZOS4`
/// semantics), `ksize` is fixed at twice the filter support (4 for bicubic,
/// 6 for lanczos) regardless of scale. Much faster on strong downscale but
/// aliases.
#[allow(clippy::needless_range_loop)]
pub(super) fn precompute_contribs(
    src_size: usize,
    dst_size: usize,
    filt: FilterKind,
    antialias: bool,
) -> (Vec<i32>, Vec<i32>, usize) {
    const Q: i32 = 14;
    const SCALE: i32 = 1 << Q;
    let scale = src_size as f64 / dst_size as f64;
    let filt_scale = if antialias { scale.max(1.0) } else { 1.0 };
    let support = filt.support() * filt_scale;
    let ksize = ((support.ceil() as usize) * 2).max(2);

    let mut offsets = vec![0i32; dst_size];
    let mut weights = vec![0i32; dst_size * ksize];

    for i in 0..dst_size {
        let center = (i as f64 + 0.5) * scale - 0.5;
        let left = (center - support).ceil() as i64;
        offsets[i] = left as i32;

        let inv_filt_scale = 1.0 / filt_scale;
        let mut raw = vec![0f64; ksize];
        let mut sum = 0f64;
        for k in 0..ksize {
            let x = (left + k as i64) as f64 - center;
            let w = filt.weight(x * inv_filt_scale) * inv_filt_scale;
            raw[k] = w;
            sum += w;
        }
        let mut qw = vec![0i32; ksize];
        let mut qsum = 0i32;
        let norm = if sum.abs() > 1e-12 {
            SCALE as f64 / sum
        } else {
            0.0
        };
        for k in 0..ksize {
            let v = (raw[k] * norm).round() as i32;
            qw[k] = v;
            qsum += v;
        }
        if qsum != SCALE {
            let mut max_k = 0usize;
            let mut max_abs = 0i32;
            for k in 0..ksize {
                if qw[k].abs() > max_abs {
                    max_abs = qw[k].abs();
                    max_k = k;
                }
            }
            qw[max_k] += SCALE - qsum;
        }
        weights[i * ksize..(i + 1) * ksize].copy_from_slice(&qw);
    }
    (offsets, weights, ksize)
}

/// Compact per-tap LUT: `xsrc` as `u16`, clamped to `[0, src_w-1]`.
///
/// At `kx=30` and `dst_w=224` the combined `(xsrc, xw)` table fits L1 (27 KB)
/// where `u32` / `i32` would spill (54 KB).
#[allow(clippy::needless_range_loop)]
pub(super) fn build_xsrc_lut(xofs: &[i32], dst_w: usize, kx: usize, src_w: usize) -> Vec<u16> {
    debug_assert!(src_w <= u16::MAX as usize);
    let mut xsrc = Vec::<u16>::with_capacity(dst_w * kx);
    for x in 0..dst_w {
        let x0 = xofs[x];
        for t in 0..kx {
            let sx = (x0 + t as i32).clamp(0, src_w as i32 - 1) as u16;
            xsrc.push(sx);
        }
    }
    xsrc
}

/// Pack Q14 weights `i32 → i16`. Coefficients have ≤ 16384 peak magnitude and
/// fit `i16` with headroom; halves LUT bandwidth and lets the horizontal inner
/// loop issue `vmlal_n_s16` directly without per-iter `as i16` casts.
pub(super) fn pack_xw_i16(xw: &[i32]) -> Vec<i16> {
    xw.iter().map(|&w| w as i16).collect()
}
