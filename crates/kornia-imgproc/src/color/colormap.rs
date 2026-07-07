use kornia_image::{Image, ImageError};

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
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
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

// Single source of truth: (lowercase name, variant, LUT).
// `from_name` and `lut` both derive from this table, so adding a new colormap
// requires editing exactly one place instead of two parallel 21-arm matches.
static COLORMAPS: &[(&str, ColormapType, &ColormapLut)] = &[
    ("autumn", ColormapType::Autumn, &AUTUMN_LUT),
    ("bone", ColormapType::Bone, &BONE_LUT),
    ("jet", ColormapType::Jet, &JET_LUT),
    ("winter", ColormapType::Winter, &WINTER_LUT),
    ("rainbow", ColormapType::Rainbow, &RAINBOW_LUT),
    ("ocean", ColormapType::Ocean, &OCEAN_LUT),
    ("summer", ColormapType::Summer, &SUMMER_LUT),
    ("spring", ColormapType::Spring, &SPRING_LUT),
    ("cool", ColormapType::Cool, &COOL_LUT),
    ("hsv", ColormapType::Hsv, &HSV_LUT),
    ("pink", ColormapType::Pink, &PINK_LUT),
    ("hot", ColormapType::Hot, &HOT_LUT),
    ("parula", ColormapType::Parula, &PARULA_LUT),
    ("magma", ColormapType::Magma, &MAGMA_LUT),
    ("inferno", ColormapType::Inferno, &INFERNO_LUT),
    ("plasma", ColormapType::Plasma, &PLASMA_LUT),
    ("viridis", ColormapType::Viridis, &VIRIDIS_LUT),
    ("cividis", ColormapType::Cividis, &CIVIDIS_LUT),
    ("twilight", ColormapType::Twilight, &TWILIGHT_LUT),
    ("turbo", ColormapType::Turbo, &TURBO_LUT),
    ("deepgreen", ColormapType::Deepgreen, &DEEPGREEN_LUT),
];

impl ColormapType {
    /// Parse a colormap name (case-insensitive, matches OpenCV lowercase convention).
    ///
    /// Valid names: `autumn`, `bone`, `jet`, `winter`, `rainbow`, `ocean`, `summer`,
    /// `spring`, `cool`, `hsv`, `pink`, `hot`, `parula`, `magma`, `inferno`, `plasma`,
    /// `viridis`, `cividis`, `twilight`, `turbo`, `deepgreen`.
    pub fn from_name(name: &str) -> Option<Self> {
        COLORMAPS
            .iter()
            .find(|(n, _, _)| n.eq_ignore_ascii_case(name))
            .map(|&(_, variant, _)| variant)
    }

    pub(crate) fn lut(self) -> &'static ColormapLut {
        COLORMAPS
            .iter()
            .find(|(_, v, _)| *v == self)
            .map(|&(_, _, lut)| lut)
            .expect("all ColormapType variants are in COLORMAPS")
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

/// Look up 16 indices in a 256-entry single-channel LUT using NEON table lookup.
///
/// # Safety
/// `p` must point to a 256-byte array aligned to at least 1 byte.
/// `idx`, `idx64`, `idx128`, `idx192` are the raw index vector and three
/// offset-subtracted variants for the upper 64-entry chunks.
#[cfg(target_arch = "aarch64")]
#[inline]
unsafe fn lookup_channel(
    p: *const u8,
    idx: std::arch::aarch64::uint8x16_t,
    idx64: std::arch::aarch64::uint8x16_t,
    idx128: std::arch::aarch64::uint8x16_t,
    idx192: std::arch::aarch64::uint8x16_t,
) -> std::arch::aarch64::uint8x16_t {
    unsafe {
        use std::arch::aarch64::*;
        // Each vqtbl4q_u8 covers one 64-entry chunk; out-of-range indices produce 0,
        // so OR-ing the four results gives the correct value for all 256 indices.
        vorrq_u8(
            vorrq_u8(
                vqtbl4q_u8(vld1q_u8_x4(p), idx),
                vqtbl4q_u8(vld1q_u8_x4(p.add(64)), idx64),
            ),
            vorrq_u8(
                vqtbl4q_u8(vld1q_u8_x4(p.add(128)), idx128),
                vqtbl4q_u8(vld1q_u8_x4(p.add(192)), idx192),
            ),
        )
    }
}

/// NEON kernel (aarch64): 16 pixels/iteration using `vqtbl4q_u8` + `vst3q_u8`.
///
/// Each 256-byte channel LUT is split into four 64-byte chunks. `vqtbl4q_u8`
/// returns 0 for indices ≥ 64, so OR-ing four offset-adjusted lookups gives
/// the correct result for all 256 indices. The LUT is 768 bytes total and
/// stays resident in L1 after the first access.
///
/// # Safety
/// Caller must ensure `dst.len() >= src.len() * 3`.
#[cfg(target_arch = "aarch64")]
unsafe fn apply_neon(src: &[u8], dst: &mut [u8], lut: &ColormapLut) {
    unsafe {
        use std::arch::aarch64::*;

        let off64 = vdupq_n_u8(64);
        let off128 = vdupq_n_u8(128);
        let off192 = vdupq_n_u8(192);

        let n = src.len();
        let mut si = 0usize;
        let mut di = 0usize;

        while si + 16 <= n {
            let idx = vld1q_u8(src.as_ptr().add(si));
            let idx64 = vsubq_u8(idx, off64);
            let idx128 = vsubq_u8(idx, off128);
            let idx192 = vsubq_u8(idx, off192);

            let r = lookup_channel(lut.r.as_ptr(), idx, idx64, idx128, idx192);
            let g = lookup_channel(lut.g.as_ptr(), idx, idx64, idx128, idx192);
            let b = lookup_channel(lut.b.as_ptr(), idx, idx64, idx128, idx192);

            vst3q_u8(dst.as_mut_ptr().add(di), uint8x16x3_t(r, g, b));

            si += 16;
            di += 48;
        }

        // Scalar tail for remaining pixels
        apply_scalar(&src[si..], &mut dst[di..], lut);
    }
}

/// AVX2 kernel (x86_64): 8 pixels/iteration via a 32-bit gather over a packed
/// RGBA LUT.
///
/// AVX2's `_mm256_shuffle_epi8` is a 16-entry in-lane table, too small for the
/// 256-entry colormap. Instead we pack the three channel LUTs into a single
/// 256-entry `u32` table (`0x00BBGGRR` per index) and `_mm256_i32gather_epi32`
/// eight RGBA words per iteration. The interleaved RGB store is done scalar from
/// the gathered words (a full SIMD RGBA→RGB repack buys little for this
/// gather-bound kernel).
///
/// # Safety
/// Caller must ensure AVX2 is available and `dst.len() >= src.len() * 3`.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn apply_avx2(src: &[u8], dst: &mut [u8], lut: &ColormapLut) {
    use std::arch::x86_64::*;

    // Pack the 3 channel LUTs into one 256-entry RGBA word table: 0x00BBGGRR.
    let mut packed = [0u32; 256];
    for (i, w) in packed.iter_mut().enumerate() {
        *w = (lut.r[i] as u32) | ((lut.g[i] as u32) << 8) | ((lut.b[i] as u32) << 16);
    }
    let base = packed.as_ptr() as *const i32;

    let n = src.len();
    let mut si = 0usize;
    let mut di = 0usize;

    while si + 8 <= n {
        // Load 8 indices (u8) and zero-extend to 8×i32.
        let idx8 = _mm_loadl_epi64(src.as_ptr().add(si) as *const __m128i);
        let idx32 = _mm256_cvtepu8_epi32(idx8);
        // Gather 8 RGBA words: packed[idx]. scale=4 (u32 stride).
        let rgba = _mm256_i32gather_epi32::<4>(base, idx32);

        // Store gathered words and emit interleaved RGB (drop the alpha byte).
        let mut words = [0u32; 8];
        _mm256_storeu_si256(words.as_mut_ptr() as *mut __m256i, rgba);
        for &w in words.iter() {
            *dst.as_mut_ptr().add(di) = w as u8;
            *dst.as_mut_ptr().add(di + 1) = (w >> 8) as u8;
            *dst.as_mut_ptr().add(di + 2) = (w >> 16) as u8;
            di += 3;
        }
        si += 8;
    }

    // Scalar tail.
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
pub fn apply_colormap(
    src: &Image<u8, 1>,
    dst: &mut Image<u8, 3>,
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
    #[cfg(feature = "cuda")]
    {
        use super::cuda_dispatch::{pair_residency, Residency};
        if let Residency::Device(exec) = pair_residency(src, dst)? {
            return exec.run(|stream| {
                super::cuda_dispatch::apply_colormap_u8_cuda(src, dst, colormap, stream)
            });
        }
    }
    let lut = colormap.lut();

    #[cfg(target_arch = "aarch64")]
    {
        // SAFETY: NEON is baseline on aarch64; size check passed above.
        unsafe { apply_neon(src.as_slice(), dst.as_slice_mut(), lut) };
        return Ok(());
    }

    #[cfg(target_arch = "x86_64")]
    {
        if crate::simd::cpu_features().has_avx2 {
            // SAFETY: AVX2 confirmed by the runtime probe; size check passed.
            unsafe { apply_avx2(src.as_slice(), dst.as_slice_mut(), lut) };
            return Ok(());
        }
    }

    #[allow(unreachable_code)]
    {
        apply_scalar(src.as_slice(), dst.as_slice_mut(), lut);
        Ok(())
    }
}
