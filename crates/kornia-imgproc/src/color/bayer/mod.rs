//! Bayer mosaic demosaicing (u8, bilinear).
//!
//! Reconstructs a 3-channel RGB image from a single-channel Bayer mosaic using
//! rounded-integer bilinear interpolation, bit-compatible with OpenCV's interior
//! result. See [`kernels`] for the per-cell interpolation rules.

use kornia_image::{
    color_spaces::{Bayer8, BayerPattern},
    Image, ImageError,
};

pub(crate) mod kernels;

/// Demosaic a single-channel Bayer mosaic into an interleaved RGB u8 image.
///
/// `pattern` names the color layout of the 2×2 sensor cell (see [`BayerPattern`]).
/// Interpolation is rounded-integer bilinear with replicate-border addressing,
/// matching OpenCV's interior output bit-for-bit.
///
/// Note the OpenCV naming offset: kornia [`BayerPattern::Rggb`] corresponds to
/// `cv2.COLOR_BayerBG2RGB`.
///
/// # Errors
/// Returns [`ImageError::InvalidImageSize`] if `src` and `dst` differ in size.
///
/// # Example
/// ```
/// use kornia_image::{Image, ImageSize};
/// use kornia_image::color_spaces::BayerPattern;
/// use kornia_imgproc::color::rgb_from_bayer;
///
/// let mosaic = Image::<u8, 1>::from_size_val(
///     ImageSize { width: 4, height: 4 }, 128).unwrap();
/// let mut rgb = Image::<u8, 3>::from_size_val(mosaic.size(), 0).unwrap();
/// rgb_from_bayer(&mosaic, BayerPattern::Rggb, &mut rgb).unwrap();
/// ```
pub fn rgb_from_bayer(
    src: &Image<u8, 1>,
    pattern: BayerPattern,
    dst: &mut Image<u8, 3>,
) -> Result<(), ImageError> {
    if src.size() != dst.size() {
        return Err(ImageError::InvalidImageSize(
            src.cols(),
            src.rows(),
            dst.cols(),
            dst.rows(),
        ));
    }
    #[cfg(feature = "cuda")]
    {
        use super::cuda_dispatch::{pair_residency, Residency};
        if let Residency::Device(exec) = pair_residency(src, dst)? {
            return exec.run(|stream| {
                super::cuda_dispatch::rgb_from_bayer_u8_cuda(src, dst, pattern, stream)
            });
        }
    }
    kernels::rgb_from_bayer_dispatch(
        src.as_slice(),
        dst.as_slice_mut(),
        src.rows(),
        src.cols(),
        pattern,
    );
    Ok(())
}

/// Demosaic a [`Bayer8`] mosaic (which carries its own pattern) into RGB u8.
///
/// Convenience wrapper around [`rgb_from_bayer`] using `src.pattern`.
///
/// # Errors
/// Returns [`ImageError::InvalidImageSize`] if `src` and `dst` differ in size.
pub fn rgb_from_bayer8(src: &Bayer8, dst: &mut Image<u8, 3>) -> Result<(), ImageError> {
    rgb_from_bayer(src.as_image(), src.pattern, dst)
}

#[cfg(test)]
mod tests {
    use super::*;
    use kornia_image::ImageSize;

    const PATTERNS: [BayerPattern; 4] = [
        BayerPattern::Rggb,
        BayerPattern::Bggr,
        BayerPattern::Grbg,
        BayerPattern::Gbrg,
    ];

    fn demosaic(data: &[u8], w: usize, h: usize, p: BayerPattern) -> Vec<u8> {
        let src = Image::<u8, 1>::new(
            ImageSize {
                width: w,
                height: h,
            },
            data.to_vec(),
        )
        .unwrap();
        let mut dst = Image::<u8, 3>::from_size_val(src.size(), 0).unwrap();
        rgb_from_bayer(&src, p, &mut dst).unwrap();
        dst.as_slice().to_vec()
    }

    /// SIMD/dispatch output must equal the scalar oracle bit-for-bit, for all
    /// four patterns and several sizes (incl. odd dims and ones that span the
    /// 16-wide NEON block plus a scalar remainder).
    #[test]
    fn neon_matches_scalar_all_patterns() {
        for &(w, h) in &[(4, 4), (5, 5), (7, 6), (33, 4), (40, 9), (3, 3), (2, 2)] {
            let data: Vec<u8> = (0..w * h).map(|i| ((i * 37 + 11) % 256) as u8).collect();
            for &p in &PATTERNS {
                let mut oracle = vec![0u8; w * h * 3];
                kernels::rgb_from_bayer_scalar(&data, &mut oracle, h, w, p);
                let got = demosaic(&data, w, h, p);
                assert_eq!(got, oracle, "pattern {p:?} size {w}x{h}");
            }
        }
    }

    /// A flat mosaic must produce a flat gray RGB image (all interpolations of a
    /// constant return that constant).
    #[test]
    fn flat_mosaic_is_flat() {
        for &p in &PATTERNS {
            let out = demosaic(&[200u8; 6 * 6], 6, 6, p);
            assert!(out.iter().all(|&v| v == 200), "pattern {p:?}");
        }
    }

    /// Hand-checked RGGB interior pixel. 4×4 ramp; verify the center R-sensel at
    /// (1,1) reconstructs G/B from its rounded neighbor averages.
    #[test]
    fn rggb_interior_known_value() {
        // 4x4 mosaic, RGGB. Phase: (r,c) even/even = R.
        // Use a small known pattern so we can compute by hand.
        #[rustfmt::skip]
        let data: [u8; 16] = [
            10, 20, 30, 40,
            50, 60, 70, 80,
            90,100,110,120,
           130,140,150,160,
        ];
        let out = demosaic(&data, 4, 4, BayerPattern::Rggb);
        // Pixel (1,1) value 60: in RGGB, (1,1) is a B sensel (row1,col1 -> B).
        // G = avg4(N=20,S=100,W=50,E=70) = (240+2)>>2 = 60
        // R = avg4(NW=10,NE=30,SW=90,SE=110) = (240+2)>>2 = 60
        let o = (4 + 1) * 3;
        assert_eq!(out[o], 60, "R@(1,1)");
        assert_eq!(out[o + 1], 60, "G@(1,1)");
        assert_eq!(out[o + 2], 60, "B@(1,1) = center");

        // Pixel (1,2) value 70: RGGB (row1,col2) -> G on B row.
        // B(horizontal) = avg2(W=60,E=80) = 70 ; R(vertical) = avg2(N=30,S=110) = 70
        let o2 = (4 + 2) * 3;
        assert_eq!(out[o2], 70, "R@(1,2)");
        assert_eq!(out[o2 + 1], 70, "G@(1,2) = center");
        assert_eq!(out[o2 + 2], 70, "B@(1,2)");
    }

    /// Corner pixels exercise replicate-border addressing.
    #[test]
    fn corners_use_replicate_border() {
        // 4x4 RGGB, (0,0) is R. With replicate border all out-of-bounds
        // neighbors clamp to in-bounds edge values.
        #[rustfmt::skip]
        let data: [u8; 16] = [
            10, 20, 30, 40,
            50, 60, 70, 80,
            90,100,110,120,
           130,140,150,160,
        ];
        let out = demosaic(&data, 4, 4, BayerPattern::Rggb);
        // (0,0) R-sensel value 10.
        // G = avg4(N=10(clamp),S=50,W=10(clamp),E=20) = (90+2)>>2 = 23
        // B = avg4(NW=10,NE=20,SW=50,SE=60) (clamped) = (140+2)>>2 = 35
        assert_eq!(out[0], 10, "R@(0,0)");
        assert_eq!(out[1], 23, "G@(0,0)");
        assert_eq!(out[2], 35, "B@(0,0)");
    }

    #[test]
    fn size_mismatch_errors() {
        let src = Image::<u8, 1>::from_size_val(
            ImageSize {
                width: 4,
                height: 4,
            },
            0,
        )
        .unwrap();
        let mut dst = Image::<u8, 3>::from_size_val(
            ImageSize {
                width: 5,
                height: 4,
            },
            0,
        )
        .unwrap();
        assert!(rgb_from_bayer(&src, BayerPattern::Rggb, &mut dst).is_err());
    }
}
