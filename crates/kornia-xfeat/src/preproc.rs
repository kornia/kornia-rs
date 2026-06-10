//! Image preprocessing helpers — getting an arbitrary `kornia_image::Image`
//! into the f32 grayscale `(H', W')` that XFeat expects.
//!
//! Steps performed (matching upstream `Xfeat.preprocess_tensor` +
//! `parse_input`):
//!
//! 1. Convert input to f32 in `[0, 1]`.
//! 2. Reduce 3-channel inputs to gray via simple channel mean
//!    (`x.mean(dim=1)`) — NOT Rec.709 weighting. The upstream model expects
//!    this exact reduction so we must match.
//! 3. Round H and W down to the nearest multiple of 32; record the spatial
//!    scale ratios so post-proc can rescale keypoints back to the original.
//! 4. Bilinear resample to the rounded `(H', W')`.
//!
//! `InstanceNorm2d(1)` is **not** here — that's done inside the model
//! `forward` so callers don't have to know about it.

/// Spatial scale ratios used to map normalized-resolution keypoints back to
/// the caller's original image coordinates: `x_orig = x_norm * rw`,
/// `y_orig = y_norm * rh`.
#[derive(Debug, Clone, Copy)]
pub struct InputScale {
    /// Width ratio `w_orig / w_norm`.
    pub rw: f32,
    /// Height ratio `h_orig / h_norm`.
    pub rh: f32,
    /// Width fed to the model (multiple of 32).
    pub w_norm: usize,
    /// Height fed to the model (multiple of 32).
    pub h_norm: usize,
}

/// Round H and W down to the nearest multiple of 32 (XFeat's stride alignment).
pub fn align_to_32(h: usize, w: usize) -> (usize, usize) {
    ((h / 32) * 32, (w / 32) * 32)
}

/// Bilinear resample a gray f32 image from `(h_in, w_in)` to `(h_out, w_out)`.
/// Same convention as `F.interpolate(..., mode='bilinear', align_corners=False)`.
pub fn bilinear_resample_gray(
    src: &[f32],
    dst: &mut [f32],
    h_in: usize,
    w_in: usize,
    h_out: usize,
    w_out: usize,
) {
    assert_eq!(src.len(), h_in * w_in);
    assert_eq!(dst.len(), h_out * w_out);
    let sh = h_in as f32 / h_out as f32;
    let sw = w_in as f32 / w_out as f32;
    for oh in 0..h_out {
        let ys = (oh as f32 + 0.5) * sh - 0.5;
        let y0 = ys.floor();
        let wy = ys - y0;
        let y0i = (y0 as isize).clamp(0, h_in as isize - 1) as usize;
        let y1i = (y0i + 1).min(h_in - 1);
        for ow in 0..w_out {
            let xs = (ow as f32 + 0.5) * sw - 0.5;
            let x0 = xs.floor();
            let wx = xs - x0;
            let x0i = (x0 as isize).clamp(0, w_in as isize - 1) as usize;
            let x1i = (x0i + 1).min(w_in - 1);
            let v00 = src[y0i * w_in + x0i];
            let v01 = src[y0i * w_in + x1i];
            let v10 = src[y1i * w_in + x0i];
            let v11 = src[y1i * w_in + x1i];
            let v0 = v00 * (1.0 - wx) + v01 * wx;
            let v1 = v10 * (1.0 - wx) + v11 * wx;
            dst[oh * w_out + ow] = v0 * (1.0 - wy) + v1 * wy;
        }
    }
}

/// Convert an interleaved RGB u8 image (`H*W*3` bytes) to f32 gray in `[0, 1]`
/// via channel mean.
pub fn rgb_u8_to_gray_f32(src: &[u8], dst: &mut [f32], h: usize, w: usize) {
    assert_eq!(src.len(), h * w * 3);
    assert_eq!(dst.len(), h * w);
    let inv255 = 1.0 / 255.0;
    for px in 0..(h * w) {
        let r = src[px * 3] as f32;
        let g = src[px * 3 + 1] as f32;
        let b = src[px * 3 + 2] as f32;
        dst[px] = (r + g + b) / 3.0 * inv255;
    }
}

/// Convert a gray u8 image (`H*W` bytes) to f32 gray in `[0, 1]`.
pub fn gray_u8_to_gray_f32(src: &[u8], dst: &mut [f32]) {
    assert_eq!(src.len(), dst.len());
    let inv255 = 1.0 / 255.0;
    for (d, &s) in dst.iter_mut().zip(src) {
        *d = s as f32 * inv255;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn align_to_32_rounds_down() {
        assert_eq!(align_to_32(480, 640), (480, 640));
        assert_eq!(align_to_32(481, 641), (480, 640));
        assert_eq!(align_to_32(479, 639), (448, 608));
    }

    #[test]
    fn bilinear_resample_identity() {
        let src = vec![1.0, 2.0, 3.0, 4.0];
        let mut dst = vec![0.0; 4];
        bilinear_resample_gray(&src, &mut dst, 2, 2, 2, 2);
        for (a, b) in src.iter().zip(dst.iter()) {
            assert!((a - b).abs() < 1e-5);
        }
    }
}
