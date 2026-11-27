//! Single-channel u8 bilinear remap through a precomputed table.
//!
//! [`super::remap_u8`] floors and splits every f32 map entry on every call,
//! one scalar pixel at a time. A rectification or undistortion map is
//! constant, so [`RemapTable`] computes the integer taps and fixed-point
//! weights once, and [`remap_u8_table`] blends on `wide` lanes, on the calling
//! thread, without allocating.
//!
//! Reference: OpenCV `convertMaps` (`CV_16SC2`) + `remap` (`INTER_LINEAR`,
//! `BORDER_CONSTANT` 0). As OpenCV, each output pixel reads its two taps per
//! source row as one 16-bit load and blends them with 16-bit multiply-adds
//! (`pmaddwd`, NEON `vmlal`), which baseline SSE2 has; it has no 32-bit lane
//! multiply. [`RemapQuantisation::Q10Trunc`] instead reproduces
//! [`super::remap_u8`] byte for byte. Departures from OpenCV: a pixel whose
//! top-left tap is outside the source is 0, and a tap on the last row or
//! column is clamped to the edge pixel rather than read as border 0, as
//! [`super::remap_u8`] does.

use kornia_image::{Image, ImageError, ImageSize};
use wide::{i16x16, i16x8, i32x8, u8x16};

use crate::simd::lanes::{lane_starts, LANES};

/// How map coordinates become taps and weights.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RemapQuantisation {
    /// [`super::remap_u8`]: `floor` taps, 10-bit truncated fractions,
    /// `(Σ w·p + 2^19) >> 20`. Byte-identical to it.
    Q10Trunc,
    /// OpenCV `convertMaps` to `CV_16SC2`: coordinates rounded to 1/32 px,
    /// `(Σ w·p + 512) >> 10`. OpenCV's 15-bit weights are these times 32, so
    /// the sum rounds the same.
    OpenCvQ5,
}

impl RemapQuantisation {
    const fn bits(self) -> u32 {
        match self {
            Self::Q10Trunc => 10,
            Self::OpenCvQ5 => 5,
        }
    }
}

/// A pixel whose taps the lane loop cannot read at fixed strides: the last
/// source row, where the clamped neighbours sit on the same row.
#[derive(Debug, Clone, Copy)]
struct EdgeTap {
    dst: u32,
    taps: [u32; 4],
    fx: u16,
    fy: u16,
}

/// Taps and weights of one remap, for sources of one size.
///
/// Built once by [`RemapTable::from_maps`]; applied by [`remap_u8_table`].
#[derive(Debug, Clone)]
pub struct RemapTable {
    src_size: ImageSize,
    dst_size: ImageSize,
    bits: u32,
    /// Top-left tap per output pixel; its neighbours are `+1`, `+w`, `+w+1`.
    offsets: Vec<u32>,
    /// `[S - fx, fx]` per output pixel, `S = 2^bits`; `fx = 0` where the
    /// right tap is clamped.
    weights_x: Vec<[i16; 2]>,
    /// `[S - fy, fy]` per output pixel.
    weights_y: Vec<[i16; 2]>,
    /// Output pixels whose top-left tap is outside the source: zero.
    zeros: Vec<u32>,
    edges: Vec<EdgeTap>,
}

/// The top-left tap and fraction of one coordinate, `None` outside `0..len`.
fn tap(v: f32, len: i32, quant: RemapQuantisation) -> Option<(i32, u32)> {
    if !v.is_finite() {
        return None;
    }
    let (i, frac) = match quant {
        // The expressions of `remap_u8`, so every byte matches.
        RemapQuantisation::Q10Trunc => {
            let i = v.floor() as i32;
            (i, ((v - i as f32) * 1024.0) as u32)
        }
        RemapQuantisation::OpenCvQ5 => {
            let q = (v * 32.0).round_ties_even() as i32;
            (q >> 5, (q & 31) as u32)
        }
    };
    (0..len).contains(&i).then_some((i, frac))
}

impl RemapTable {
    /// Builds the table of `map_x`, `map_y` for sources of `src_size`.
    ///
    /// # Arguments
    ///
    /// * `map_x` - Source x coordinate (pixels) per output pixel.
    /// * `map_y` - Source y coordinate (pixels) per output pixel, same size as `map_x`.
    /// * `src_size` - Size of the images [`remap_u8_table`] will sample.
    /// * `quant` - How coordinates become taps and weights.
    ///
    /// # Returns
    ///
    /// The table, for outputs of the maps' size.
    ///
    /// # Errors
    ///
    /// [`ImageError::InvalidImageSize`] when the maps differ in size, or the
    /// source is narrower or shorter than 2 pixels or has more than `u32::MAX`
    /// pixels.
    ///
    /// # Example
    ///
    /// ```rust
    /// use kornia_image::{Image, ImageSize};
    /// use kornia_imgproc::interpolation::{remap_u8_table, RemapQuantisation, RemapTable};
    ///
    /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// let size = ImageSize { width: 16, height: 8 };
    /// // Shift right by half a pixel.
    /// let n = size.width * size.height;
    /// let map_x = Image::<f32, 1>::new(size, (0..n).map(|i| (i % 16) as f32 + 0.5).collect())?;
    /// let map_y = Image::<f32, 1>::new(size, (0..n).map(|i| (i / 16) as f32).collect())?;
    /// let table = RemapTable::from_maps(&map_x, &map_y, size, RemapQuantisation::OpenCvQ5)?;
    ///
    /// let src = Image::<u8, 1>::new(size, (0..n).map(|i| (i % 16) as u8 * 10).collect())?;
    /// let mut dst = Image::<u8, 1>::from_size_val(size, 0)?;
    /// remap_u8_table(&src, &mut dst, &table)?;
    /// assert_eq!(dst.as_slice()[0], 5);
    /// # Ok(())
    /// # }
    /// ```
    pub fn from_maps(
        map_x: &Image<f32, 1>,
        map_y: &Image<f32, 1>,
        src_size: ImageSize,
        quant: RemapQuantisation,
    ) -> Result<Self, ImageError> {
        if map_x.size() != map_y.size() {
            return Err(ImageError::InvalidImageSize(
                map_x.width(),
                map_x.height(),
                map_y.width(),
                map_y.height(),
            ));
        }
        let (w, h) = (src_size.width, src_size.height);
        if w < 2 || h < 2 || w * h > u32::MAX as usize {
            return Err(ImageError::InvalidImageSize(w, h, 2, 2));
        }
        let n = map_x.width() * map_x.height();
        let mut table = Self {
            src_size,
            dst_size: map_x.size(),
            bits: quant.bits(),
            offsets: vec![0; n],
            weights_x: vec![[1 << quant.bits(), 0]; n],
            weights_y: vec![[1 << quant.bits(), 0]; n],
            zeros: Vec::new(),
            edges: Vec::new(),
        };
        let scale = 1i16 << quant.bits();
        for (dst, (&xf, &yf)) in map_x.as_slice().iter().zip(map_y.as_slice()).enumerate() {
            let taps = tap(xf, w as i32, quant).zip(tap(yf, h as i32, quant));
            let Some(((xi, fx), (yi, fy))) = taps else {
                table.zeros.push(dst as u32);
                continue;
            };
            let (xi, yi) = (xi as usize, yi as usize);
            // A clamped neighbour is the same pixel, so its weight can move to
            // the near one: `p·(S - f) + p·f = p·S`.
            let (dx, fx) = if xi + 1 < w { (1, fx) } else { (0, 0) };
            let (dy, fy) = if yi + 1 < h { (w, fy) } else { (0, 0) };
            let (fx, fy) = (fx as u16, fy as u16);
            let off = yi * w + xi;
            // The lane loop reads `off ..= off + w + 1`; only the last row and
            // the last pixel of the row above it reach past the source.
            if off + w + 2 <= w * h {
                let pair = |f: u16| [scale - f as i16, f as i16];
                table.offsets[dst] = off as u32;
                table.weights_x[dst] = pair(fx);
                table.weights_y[dst] = pair(fy);
            } else {
                table.edges.push(EdgeTap {
                    dst: dst as u32,
                    taps: [off, off + dx, off + dy, off + dy + dx].map(|t| t as u32),
                    fx,
                    fy,
                });
            }
        }
        Ok(table)
    }
}

/// Reinterprets each i32 lane `lo | hi << 16` as the i16 pair `[lo, hi]`.
#[inline(always)]
fn as_pairs(v: i32x8) -> i16x16 {
    // SAFETY: both are 32-byte vectors of plain integers (`bytemuck::Pod` in
    // `wide`), so every bit pattern is a valid value of either. Lane order is
    // little-endian, as on x86_64 and aarch64.
    unsafe { std::mem::transmute::<i32x8, i16x16>(v) }
}

/// Eight output pixels from their tap pairs: `top`/`bottom` hold
/// `[p00, p01]` / `[p10, p11]` per pixel. The rows are `[p0, p1]·[S - fx, fx]`
/// in one multiply-add. A Q10 row reaches 2^18, past i16, so the vertical
/// step splits each row into `1024·hi + lo` and runs two multiply-adds:
/// `Σ row·w = 1024·Σ hi·w + Σ lo·w`, exact, as the reference's
/// `(top·(S - fy) + bottom·fy + S²/2) >> 2·bits`.
#[inline(always)]
fn blend8(top: u8x16, bottom: u8x16, wx: i16x16, wy: i16x16, bits: u32) -> [u8; LANES] {
    let t = i16x16::from(top).dot(wx);
    let b = i16x16::from(bottom).dot(wx);
    let low = i32x8::splat(1023);
    let hi = as_pairs((t >> 10) | ((b >> 10) << 16)).dot(wy);
    let lo = as_pairs((t & low) | ((b & low) << 16)).dot(wy);
    let v = ((hi << 10) + lo + i32x8::splat(1 << (2 * bits - 1))) >> (2 * bits) as i32;
    let bytes = u8x16::narrow_i16x8(i16x8::from_i32x8_saturate(v), i16x8::ZERO).to_array();
    std::array::from_fn(|k| bytes[k])
}

#[inline(always)]
fn blend_scalar(p: [i32; 4], fx: i32, fy: i32, bits: u32) -> u8 {
    let scale = 1 << bits;
    let top = p[0] * (scale - fx) + p[1] * fx;
    let bottom = p[2] * (scale - fx) + p[3] * fx;
    ((top * (scale - fy) + bottom * fy + (1 << (2 * bits - 1))) >> (2 * bits)) as u8
}

/// Samples `src` through `table` into `dst`, bilinearly. No allocation.
///
/// # Arguments
///
/// * `src` - Source image, of the size the table was built for.
/// * `dst` - Output image, of the size of the table's maps.
/// * `table` - Taps and weights from [`RemapTable::from_maps`].
///
/// # Returns
///
/// `Ok(())` once `dst` holds the remapped image.
///
/// # Errors
///
/// [`ImageError::InvalidImageSize`] when `src` or `dst` is not the size the
/// table was built for.
///
/// # Example
///
/// See [`RemapTable::from_maps`].
pub fn remap_u8_table(
    src: &Image<u8, 1>,
    dst: &mut Image<u8, 1>,
    table: &RemapTable,
) -> Result<(), ImageError> {
    for (have, want) in [(src.size(), table.src_size), (dst.size(), table.dst_size)] {
        if have != want {
            return Err(ImageError::InvalidImageSize(
                have.width,
                have.height,
                want.width,
                want.height,
            ));
        }
    }
    let stride = table.src_size.width;
    let src = src.as_slice();
    let out = dst.as_slice_mut();
    let n = out.len();
    let lanes = |start: usize, out: &mut [u8]| {
        let (mut top, mut bottom) = ([0u8; 16], [0u8; 16]);
        for k in 0..LANES {
            let off = table.offsets[start + k] as usize;
            // SAFETY: `from_maps` keeps `off + stride + 2 <= w·h` for every
            // lane pixel (the rest are fixed up after this pass), and `src`
            // was checked to be the table's `w × h`.
            let (t, b) = unsafe {
                let p = src.as_ptr().add(off);
                (
                    p.cast::<[u8; 2]>().read_unaligned(),
                    p.add(stride).cast::<[u8; 2]>().read_unaligned(),
                )
            };
            top[2 * k..2 * k + 2].copy_from_slice(&t);
            bottom[2 * k..2 * k + 2].copy_from_slice(&b);
        }
        let weights = |w: &[[i16; 2]]| {
            let mut a = [0i16; 2 * LANES];
            a.copy_from_slice(w[start..start + LANES].as_flattened());
            i16x16::new(a)
        };
        let v = blend8(
            u8x16::new(top),
            u8x16::new(bottom),
            weights(&table.weights_x),
            weights(&table.weights_y),
            table.bits,
        );
        out[start..start + LANES].copy_from_slice(&v);
    };
    if n < LANES {
        for (i, o) in out.iter_mut().enumerate() {
            let off = table.offsets[i] as usize;
            let (fx, fy) = (table.weights_x[i][1], table.weights_y[i][1]);
            let p = [off, off + 1, off + stride, off + stride + 1].map(|t| i32::from(src[t]));
            *o = blend_scalar(p, i32::from(fx), i32::from(fy), table.bits);
        }
    }
    for start in lane_starts(n) {
        lanes(start, out);
    }
    for e in &table.edges {
        let p = e.taps.map(|t| i32::from(src[t as usize]));
        out[e.dst as usize] = blend_scalar(p, i32::from(e.fx), i32::from(e.fy), table.bits);
    }
    for &z in &table.zeros {
        out[z as usize] = 0;
    }
    Ok(())
}

#[cfg(test)]
pub(crate) mod tests {
    use crate::calibration::distortion::{
        generate_correction_map_polynomial, PolynomialDistortion,
    };
    use crate::calibration::{CameraExtrinsic, CameraIntrinsic};
    use crate::interpolation::{remap_u8, InterpolationMode};

    use super::*;

    /// Undistortion maps of EuRoC MH_01 `cam0` and `cam1` (`mav0/cam*/sensor.yaml`,
    /// radtan, k1 −0.283), the radial and tangential terms a rectifier's maps carry.
    pub(crate) fn euroc_maps() -> [(Image<f32, 1>, Image<f32, 1>); 2] {
        let size = ImageSize {
            width: 752,
            height: 480,
        };
        let maps = |fx, fy, cx, cy, k1, k2, p1, p2| {
            let intrinsic = CameraIntrinsic { fx, fy, cx, cy };
            let distortion = PolynomialDistortion {
                k1,
                k2,
                k3: 0.0,
                k4: 0.0,
                k5: 0.0,
                k6: 0.0,
                p1,
                p2,
            };
            let extrinsic = CameraExtrinsic {
                rotation: [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
                translation: [0.0; 3],
            };
            generate_correction_map_polynomial(
                &intrinsic,
                &extrinsic,
                &intrinsic,
                &distortion,
                &size,
            )
            .unwrap()
        };
        [
            maps(
                458.654,
                457.296,
                367.215,
                248.375,
                -0.28340811,
                0.07395907,
                0.00019359,
                1.76187114e-05,
            ),
            maps(
                457.587,
                456.134,
                379.999,
                255.238,
                -0.28368365,
                0.07451284,
                -0.00010473,
                -3.55590700e-05,
            ),
        ]
    }

    /// A textured image with sharp steps, so every tap and weight matters.
    pub(crate) fn texture(size: ImageSize, seed: u32) -> Image<u8, 1> {
        let data = (0..size.width * size.height)
            .map(|i| {
                let (x, y) = ((i % size.width) as u32, (i / size.width) as u32);
                let h = (x.wrapping_mul(73_856_093) ^ y.wrapping_mul(19_349_663) ^ seed)
                    .wrapping_mul(2_654_435_761);
                if (x / 7 + y / 5) % 2 == 0 {
                    (h >> 24) as u8
                } else {
                    255 - ((h >> 26) as u8)
                }
            })
            .collect();
        Image::new(size, data).unwrap()
    }

    fn reference(src: &Image<u8, 1>, map_x: &Image<f32, 1>, map_y: &Image<f32, 1>) -> Vec<u8> {
        let mut dst = Image::from_size_val(map_x.size(), 0).unwrap();
        remap_u8(src, &mut dst, map_x, map_y, InterpolationMode::Bilinear).unwrap();
        dst.as_slice().to_vec()
    }

    fn ours(
        src: &Image<u8, 1>,
        map_x: &Image<f32, 1>,
        map_y: &Image<f32, 1>,
        quant: RemapQuantisation,
    ) -> Vec<u8> {
        let table = RemapTable::from_maps(map_x, map_y, src.size(), quant).unwrap();
        let mut dst = Image::from_size_val(map_x.size(), 0).unwrap();
        remap_u8_table(src, &mut dst, &table).unwrap();
        dst.as_slice().to_vec()
    }

    #[test]
    fn q10_equals_remap_u8_on_the_euroc_maps() {
        let size = ImageSize {
            width: 752,
            height: 480,
        };
        for (seed, (map_x, map_y)) in euroc_maps().iter().enumerate() {
            let src = texture(size, seed as u32);
            assert_eq!(
                ours(&src, map_x, map_y, RemapQuantisation::Q10Trunc),
                reference(&src, map_x, map_y)
            );
        }
    }

    /// NaN, infinite, negative, exact-integer, last-row and last-column taps,
    /// the open `(w - 1, w)` band, and widths that are not a multiple of 8.
    #[test]
    fn q10_equals_remap_u8_on_the_edge_taps() {
        let src = texture(
            ImageSize {
                width: 13,
                height: 9,
            },
            7,
        );
        let (w, h) = (src.width() as f32, src.height() as f32);
        let specials = [
            f32::NAN,
            f32::INFINITY,
            f32::NEG_INFINITY,
            -1e-6,
            -0.5,
            -1.0,
            0.0,
            0.25,
            1.0,
            5.0,
            w - 2.0,
            w - 1.5,
            w - 1.0,
            w - 0.25,
            w - 1e-4,
            w,
            w + 0.5,
            h - 2.0,
            h - 1.0,
            h - 0.3,
            h,
            1e9,
            -1e9,
        ];
        // Every (x, y) pair, laid out at several output widths.
        let pairs: Vec<(f32, f32)> = specials
            .iter()
            .flat_map(|&y| specials.iter().map(move |&x| (x, y)))
            .collect();
        for dst_w in [1usize, 7, 8, 11, 23] {
            let size = ImageSize {
                width: dst_w,
                height: pairs.len().div_ceil(dst_w),
            };
            let at = |i: usize| pairs[i % pairs.len()];
            let n = size.width * size.height;
            let mx = Image::new(size, (0..n).map(|i| at(i).0).collect()).unwrap();
            let my = Image::new(size, (0..n).map(|i| at(i).1).collect()).unwrap();
            assert_eq!(
                ours(&src, &mx, &my, RemapQuantisation::Q10Trunc),
                reference(&src, &mx, &my),
                "dst width {dst_w}"
            );
        }
    }

    #[test]
    fn q5_samples_exact_positions_exactly() {
        let src = texture(
            ImageSize {
                width: 20,
                height: 10,
            },
            3,
        );
        let size = ImageSize {
            width: 18,
            height: 8,
        };
        let n = size.width * size.height;
        let mx = Image::new(size, (0..n).map(|i| (i % 18) as f32 + 0.5).collect()).unwrap();
        let my = Image::new(size, (0..n).map(|i| (i / 18) as f32).collect()).unwrap();
        let out = ours(&src, &mx, &my, RemapQuantisation::OpenCvQ5);
        for (i, &v) in out.iter().enumerate() {
            let (x, y) = (i % 18, i / 18);
            let s = |x: usize| u32::from(src.as_slice()[y * 20 + x]);
            assert_eq!(u32::from(v), (s(x) * 16 + s(x + 1) * 16 + 16) >> 5);
        }
    }

    #[test]
    fn wrong_sizes_are_rejected() {
        let size = ImageSize {
            width: 4,
            height: 4,
        };
        let map = Image::from_size_val(size, 1.0f32).unwrap();
        let table = RemapTable::from_maps(&map, &map, size, RemapQuantisation::Q10Trunc).unwrap();
        let src = Image::from_size_val(size, 9u8).unwrap();
        let mut small = Image::from_size_val(
            ImageSize {
                width: 3,
                height: 4,
            },
            0u8,
        )
        .unwrap();
        assert!(remap_u8_table(&src, &mut small, &table).is_err());
        assert!(remap_u8_table(&small.clone(), &mut src.clone(), &table).is_err());
        let one_row = ImageSize {
            width: 4,
            height: 1,
        };
        assert!(RemapTable::from_maps(&map, &map, one_row, RemapQuantisation::Q10Trunc).is_err());
    }
}
