//! Runtime color-space vocabulary shared by Rust and Python.

/// A per-pixel color space. The shared vocabulary for the high-level
/// conversion API (`.cvt()` typed path and `cvt_color` runtime path).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum ColorSpace {
    /// Red-Green-Blue (3 channels, u8 or f32).
    Rgb,
    /// Blue-Green-Red (3 channels, OpenCV native order).
    Bgr,
    /// Grayscale luminance (1 channel).
    Gray,
    /// Red-Green-Blue-Alpha (4 channels).
    Rgba,
    /// Blue-Green-Red-Alpha (4 channels).
    Bgra,
    /// Hue-Saturation-Value (3 channels, f32 only).
    Hsv,
    /// Hue-Lightness-Saturation (3 channels, f32 only).
    Hls,
    /// CIE L*a*b* (3 channels, f32 only).
    Lab,
    /// CIE L*u*v* (3 channels, f32 only).
    Luv,
    /// CIE XYZ (3 channels, f32 only).
    Xyz,
    /// Linear (gamma-decoded) RGB (3 channels, f32 only).
    LinearRgb,
    /// Y′CbCr (3 channels).
    YCbCr,
    /// YUV (3 channels).
    Yuv,
}

impl ColorSpace {
    /// Number of channels an image in this space has.
    pub const fn channels(self) -> usize {
        match self {
            ColorSpace::Gray => 1,
            ColorSpace::Rgba | ColorSpace::Bgra => 4,
            _ => 3,
        }
    }

    /// True for spaces whose kernels only operate on f32 data.
    pub const fn requires_f32(self) -> bool {
        matches!(
            self,
            ColorSpace::Hsv | ColorSpace::Hls | ColorSpace::Lab
                | ColorSpace::Luv | ColorSpace::Xyz | ColorSpace::LinearRgb
        )
    }

    /// Whether a direct kernel exists for `from -> to`. Mirrors the
    /// `ConvertColor` impls in kornia-imgproc (RGB-hub graph).
    pub const fn supports(from: ColorSpace, to: ColorSpace) -> bool {
        use ColorSpace::*;
        matches!(
            (from, to),
            (Rgb, Gray) | (Gray, Rgb)
                | (Rgb, Bgr) | (Bgr, Rgb)
                | (Rgb, Rgba) | (Rgba, Rgb)
                | (Rgb, Bgra) | (Bgra, Rgb)
                | (Rgb, Hsv) | (Hsv, Rgb)
                | (Rgb, Hls) | (Hls, Rgb)
                | (Rgb, Lab) | (Lab, Rgb)
                | (Rgb, Luv) | (Luv, Rgb)
                | (Rgb, Xyz) | (Xyz, Rgb)
                | (Rgb, LinearRgb) | (LinearRgb, Rgb)
                | (Rgb, YCbCr) | (YCbCr, Rgb)
                | (Rgb, Yuv) | (Yuv, Rgb)
        )
    }
}

#[cfg(test)]
mod tests {
    use super::ColorSpace;

    #[test]
    fn channels_and_dtype_metadata() {
        assert_eq!(ColorSpace::Gray.channels(), 1);
        assert_eq!(ColorSpace::Rgb.channels(), 3);
        assert_eq!(ColorSpace::Rgba.channels(), 4);
        assert!(ColorSpace::Hsv.requires_f32());
        assert!(!ColorSpace::Gray.requires_f32());
        assert!(!ColorSpace::Bgr.requires_f32());
    }

    #[test]
    fn legality_table_matches_rgb_hub() {
        assert!(ColorSpace::supports(ColorSpace::Rgb, ColorSpace::Hsv));
        assert!(ColorSpace::supports(ColorSpace::Hsv, ColorSpace::Rgb));
        assert!(ColorSpace::supports(ColorSpace::Rgb, ColorSpace::Gray));
        // non-adjacent pair is rejected
        assert!(!ColorSpace::supports(ColorSpace::Hsv, ColorSpace::Lab));
        assert!(!ColorSpace::supports(ColorSpace::Gray, ColorSpace::Hsv));
    }
}
