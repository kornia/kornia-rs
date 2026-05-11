use kornia_image::{allocator::ImageAllocator, Image, ImageError};

// ── LUT data ──────────────────────────────────────────────────────────────────

/// Per-channel RGB lookup table for one colormap (3 × 256 bytes = 768 bytes).
pub struct ColormapLut {
    pub(crate) r: [u8; 256],
    pub(crate) g: [u8; 256],
    pub(crate) b: [u8; 256],
}

include!("colormap_luts.rs");

// ── Colormap type ─────────────────────────────────────────────────────────────

/// All 21 OpenCV colormaps, re-implemented as pure-Rust LUT tables.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ColormapType {
    /// Sequential red→yellow gradient.
    Autumn,
    /// Blue-tinted grayscale.
    Bone,
    /// Rainbow blue→cyan→green→yellow→red.
    Jet,
    /// Sequential blue→green gradient.
    Winter,
    /// Full-spectrum rainbow.
    Rainbow,
    /// Blue ocean gradient.
    Ocean,
    /// Sequential green→yellow gradient.
    Summer,
    /// Sequential magenta→yellow gradient.
    Spring,
    /// Sequential cyan→magenta gradient.
    Cool,
    /// Hue-saturation-value full hue cycle.
    Hsv,
    /// Pink/salmon tint over grayscale.
    Pink,
    /// Sequential black→red→yellow→white.
    Hot,
    /// MATLAB default perceptually uniform colormap.
    Parula,
    /// Perceptually uniform dark-purple to yellow.
    Magma,
    /// Perceptually uniform black to yellow-white.
    Inferno,
    /// Perceptually uniform blue-purple to yellow.
    Plasma,
    /// Perceptually uniform blue to yellow-green.
    Viridis,
    /// Colorblind-friendly blue-yellow diverging map.
    Cividis,
    /// Cyclic purple-white-green-black-purple.
    Twilight,
    /// High-contrast rainbow for visualization.
    Turbo,
    /// Sequential dark to bright green.
    Deepgreen,
}

impl ColormapType {
    /// Parse a colormap name (case-insensitive, matches OpenCV lowercase convention).
    pub fn from_name(name: &str) -> Option<Self> {
        match name.to_lowercase().as_str() {
            "autumn"    => Some(Self::Autumn),
            "bone"      => Some(Self::Bone),
            "jet"       => Some(Self::Jet),
            "winter"    => Some(Self::Winter),
            "rainbow"   => Some(Self::Rainbow),
            "ocean"     => Some(Self::Ocean),
            "summer"    => Some(Self::Summer),
            "spring"    => Some(Self::Spring),
            "cool"      => Some(Self::Cool),
            "hsv"       => Some(Self::Hsv),
            "pink"      => Some(Self::Pink),
            "hot"       => Some(Self::Hot),
            "parula"    => Some(Self::Parula),
            "magma"     => Some(Self::Magma),
            "inferno"   => Some(Self::Inferno),
            "plasma"    => Some(Self::Plasma),
            "viridis"   => Some(Self::Viridis),
            "cividis"   => Some(Self::Cividis),
            "twilight"  => Some(Self::Twilight),
            "turbo"     => Some(Self::Turbo),
            "deepgreen" => Some(Self::Deepgreen),
            _ => None,
        }
    }

    fn lut(self) -> &'static ColormapLut {
        match self {
            Self::Autumn    => &AUTUMN_LUT,
            Self::Bone      => &BONE_LUT,
            Self::Jet       => &JET_LUT,
            Self::Winter    => &WINTER_LUT,
            Self::Rainbow   => &RAINBOW_LUT,
            Self::Ocean     => &OCEAN_LUT,
            Self::Summer    => &SUMMER_LUT,
            Self::Spring    => &SPRING_LUT,
            Self::Cool      => &COOL_LUT,
            Self::Hsv       => &HSV_LUT,
            Self::Pink      => &PINK_LUT,
            Self::Hot       => &HOT_LUT,
            Self::Parula    => &PARULA_LUT,
            Self::Magma     => &MAGMA_LUT,
            Self::Inferno   => &INFERNO_LUT,
            Self::Plasma    => &PLASMA_LUT,
            Self::Viridis   => &VIRIDIS_LUT,
            Self::Cividis   => &CIVIDIS_LUT,
            Self::Twilight  => &TWILIGHT_LUT,
            Self::Turbo     => &TURBO_LUT,
            Self::Deepgreen => &DEEPGREEN_LUT,
        }
    }
}

// ── Kernels ───────────────────────────────────────────────────────────────────

/// Scalar kernel: 1 pixel/iteration.
fn apply_scalar(src: &[u8], dst: &mut [u8], lut: &ColormapLut) {
    for (out, &idx) in dst.chunks_exact_mut(3).zip(src.iter()) {
        let i = idx as usize;
        out[0] = lut.r[i];
        out[1] = lut.g[i];
        out[2] = lut.b[i];
    }
}

/// NEON kernel (aarch64): 16 pixels/iteration using vqtbl4q_u8 + vst3q_u8.
///
/// Each 256-byte channel LUT is split into four 64-byte chunks. `vqtbl4q_u8`
/// returns 0 for indices ≥ 64, so OR-ing four offset-adjusted lookups gives
/// the correct result for all 256 indices. The LUT is 768 bytes total and
/// stays resident in L1 after the first access.
#[cfg(target_arch = "aarch64")]
unsafe fn apply_neon(src: &[u8], dst: &mut [u8], lut: &ColormapLut) {
    use std::arch::aarch64::*;

    let off64  = vdupq_n_u8(64);
    let off128 = vdupq_n_u8(128);
    let off192 = vdupq_n_u8(192);

    let n = src.len();
    let mut si = 0usize;
    let mut di = 0usize;

    while si + 16 <= n {
        let idx = vld1q_u8(src.as_ptr().add(si));
        let idx64  = vsubq_u8(idx, off64);
        let idx128 = vsubq_u8(idx, off128);
        let idx192 = vsubq_u8(idx, off192);

        macro_rules! lookup_channel {
            ($ch:expr) => {{
                let p = $ch.as_ptr();
                vorrq_u8(
                    vorrq_u8(
                        vqtbl4q_u8(vld1q_u8_x4(p),        idx),
                        vqtbl4q_u8(vld1q_u8_x4(p.add(64)),  idx64),
                    ),
                    vorrq_u8(
                        vqtbl4q_u8(vld1q_u8_x4(p.add(128)), idx128),
                        vqtbl4q_u8(vld1q_u8_x4(p.add(192)), idx192),
                    ),
                )
            }};
        }

        let r = lookup_channel!(lut.r);
        let g = lookup_channel!(lut.g);
        let b = lookup_channel!(lut.b);

        vst3q_u8(dst.as_mut_ptr().add(di), uint8x16x3_t(r, g, b));

        si += 16;
        di += 48;
    }

    // Scalar tail for remaining pixels
    apply_scalar(&src[si..], &mut dst[di..], lut);
}

// ── Public API ────────────────────────────────────────────────────────────────

/// Apply a colormap to a single-channel u8 image, producing an RGB u8 image.
///
/// Dispatches to the NEON kernel on aarch64 (16 pixels/iter via `vqtbl4q_u8`
/// + `vst3q_u8`) and falls back to scalar on other architectures.
///
/// # Errors
/// Returns `ImageError` if `src` and `dst` sizes do not match.
pub fn apply_colormap<A: ImageAllocator>(
    src: &Image<u8, 1, A>,
    dst: &mut Image<u8, 3, A>,
    colormap: ColormapType,
) -> Result<(), ImageError> {
    if src.size() != dst.size() {
        return Err(ImageError::InvalidImageSize(
            src.cols(),
            src.rows(),
            dst.cols(),
            dst.rows(),
        ));
    }
    let lut = colormap.lut();

    #[cfg(target_arch = "aarch64")]
    {
        unsafe { apply_neon(src.as_slice(), dst.as_slice_mut(), lut) };
        return Ok(());
    }

    #[cfg(not(target_arch = "aarch64"))]
    {
        apply_scalar(src.as_slice(), dst.as_slice_mut(), lut);
        Ok(())
    }
}
