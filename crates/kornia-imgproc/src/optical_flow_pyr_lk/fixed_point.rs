//! Fixed-point pyramidal Lucas-Kanade on u8 pyramids, OpenCV's scheme.
//!
//! Images stay u8, Scharr derivatives are i16 (unscaled), bilinear weights are
//! Q14 integers and window sums are exact integers. Each level is padded once
//! (image: reflect-101, derivatives: zero) so every window reads in bounds
//! without branches. A point gets status 0 when its window leaves the padded
//! level, when the structure tensor is ill-conditioned, or when the final
//! window leaves the image, as `cv::calcOpticalFlowPyrLK` does.

use super::{PyrLKParams, PyrLKResult, TermCriteria, PYR_LK_MAX_LEVEL};
use crate::pyramid::{pyrdown_c1_u8_strided, reflect_101};
use kornia_image::{Image, ImageSize};
use rayon::prelude::*;
use thiserror::Error;

/// Fractional bits of the bilinear weights.
const W_BITS: i32 = 14;
const W_ONE: i32 = 1 << W_BITS;
/// Window samples keep 5 fractional bits (`value * 32`).
const SAMPLE_SHIFT: i32 = W_BITS - 5;
/// Scale from the integer tensor and mismatch sums to OpenCV's float ones.
const FLT_SCALE: f32 = 1.0 / (1u32 << 20) as f32;
/// Sum of `|J - I|` is in units of 1/32 pixel value.
const SAMPLE_UNIT: f32 = 32.0;
/// i16 lanes per 128-bit vector; window rows are padded to a multiple.
const LANES: usize = 8;
/// Window rows are processed in pairs (one per 128-bit half on AVX2).
const ROW_PAIR: usize = 2;
/// Two consecutive updates that cancel to within this (pixels) mean the
/// iteration oscillates; OpenCV then takes the half step and stops.
const OSCILLATION_TOL: f32 = 0.01;
/// Bound of `|J - I|` in window units: 255 * 32.
const MAX_DIFF: i64 = 255 * 32;
/// Bound of an unscaled Scharr derivative: 16 * 255.
const MAX_DERIV: i64 = 16 * 255;
/// Points per rayon task.
const POINTS_PER_TASK: usize = 16;

/// Error type for [`calc_optical_flow_pyr_lk_u8`] and [`LkPyramidU8`].
#[non_exhaustive]
#[derive(Debug, Error)]
pub enum PyrLkU8Error {
    /// The window size must be odd and at least 3.
    #[error("invalid LK window size {0}: must be odd and >= 3")]
    InvalidWindowSize(usize),
    /// `max_level` exceeds [`PYR_LK_MAX_LEVEL`].
    #[error("invalid LK max_level {0}: at most {PYR_LK_MAX_LEVEL} is supported")]
    InvalidMaxLevel(usize),
    /// The previous pyramid was built without derivatives.
    #[error("the previous pyramid has no derivatives; build it with `with_derivatives = true`")]
    MissingDerivatives,
    /// The two pyramids differ in level count, level size or window padding.
    #[error("pyramids differ: prev {prev_size:?} x {prev_levels} levels (win {prev_win}), next {next_size:?} x {next_levels} levels (win {next_win})")]
    PyramidMismatch {
        /// Level-0 size of the previous pyramid.
        prev_size: ImageSize,
        /// Level count of the previous pyramid.
        prev_levels: usize,
        /// Window size the previous pyramid was padded for.
        prev_win: usize,
        /// Level-0 size of the next pyramid.
        next_size: ImageSize,
        /// Level count of the next pyramid.
        next_levels: usize,
        /// Window size the next pyramid was padded for.
        next_win: usize,
    },
    /// `params.win_size` differs from the window the pyramids were padded for.
    #[error("params.win_size {params} differs from the pyramid window {pyramid}")]
    WindowMismatch {
        /// Window size in the parameters.
        params: usize,
        /// Window size the pyramids were built for.
        pyramid: usize,
    },
    /// Initial flow points length does not match feature count.
    #[error("next_pts_in length ({provided}) does not match prev_pts length ({expected})")]
    InitialFlowLengthMismatch {
        /// Expected number of points.
        expected: usize,
        /// Provided number of points.
        provided: usize,
    },
    /// `params.use_initial_flow` was enabled but `next_pts_in` was not provided.
    #[error("use_initial_flow is true but next_pts_in is None")]
    InitialFlowMissing,
}

/// A row-major plane with `pad` extra columns and rows on every side.
#[derive(Clone)]
struct Plane<T> {
    data: Vec<T>,
    width: usize,
    height: usize,
    pad: usize,
    stride: usize,
}

impl<T: Copy + Default> Plane<T> {
    fn new(width: usize, height: usize, pad: usize) -> Self {
        let stride = width + 2 * pad;
        Self {
            data: vec![T::default(); stride * (height + 2 * pad)],
            width,
            height,
            pad,
            stride,
        }
    }

    /// Index of pixel `(x, y)`; valid for `-pad <= x, y` and `x < width + pad`, `y < height + pad`.
    #[inline]
    fn index(&self, x: isize, y: isize) -> usize {
        ((y + self.pad as isize) * self.stride as isize + x + self.pad as isize) as usize
    }

    /// Offset of the interior origin `(0, 0)`.
    fn origin(&self) -> usize {
        self.pad * self.stride + self.pad
    }
}

impl Plane<u8> {
    /// Fills the padding by reflect-101 from the interior.
    fn reflect_border(&mut self) {
        let (w, h, pad, stride) = (self.width as i32, self.height as i32, self.pad, self.stride);
        // Source columns of the left and right padding, relative to the row start.
        let left: Vec<usize> = (0..pad as i32)
            .map(|x| pad + reflect_101(x - pad as i32, w) as usize)
            .collect();
        let right: Vec<usize> = (0..pad as i32)
            .map(|x| pad + reflect_101(w + x, w) as usize)
            .collect();
        for row in self.data[pad * stride..(pad + self.height) * stride].chunks_exact_mut(stride) {
            for (x, &src) in left.iter().enumerate() {
                row[x] = row[src];
            }
            for (x, &src) in right.iter().enumerate() {
                row[pad + self.width + x] = row[src];
            }
        }
        for y in 0..pad as i32 {
            let above = (reflect_101(y - pad as i32, h) as usize + pad) * stride;
            let below = (reflect_101(h + y, h) as usize + pad) * stride;
            self.data
                .copy_within(above..above + stride, y as usize * stride);
            let dst = (pad + h as usize + y as usize) * stride;
            self.data.copy_within(below..below + stride, dst);
        }
    }
}

/// One pyramid level: the padded image and, for the previous frame, the
/// zero-padded Scharr derivatives.
#[derive(Clone)]
struct Level {
    image: Plane<u8>,
    deriv: Option<(Plane<i16>, Plane<i16>)>,
}

/// u8 image pyramid for [`calc_optical_flow_pyr_lk_u8`], laid out as
/// `cv::buildOpticalFlowPyramid` does: every level padded by (at least) the
/// window size with reflect-101, optional i16 Scharr derivatives per level
/// (unscaled, reflect-101 inside the image, zero outside).
///
/// Build the previous frame's pyramid with derivatives and the next frame's
/// without; a pyramid built with derivatives can serve as either, so the
/// next frame's pyramid of one call can be the previous one of the next call
/// when built with derivatives.
#[derive(Clone)]
pub struct LkPyramidU8 {
    levels: Vec<Level>,
    win_size: usize,
}

/// Geometry of one tracking window: `win` real rows/columns, padded to
/// `cols` (multiple of [`LANES`]) and `rows` (multiple of [`ROW_PAIR`]).
#[derive(Clone, Copy)]
struct WinShape {
    win: usize,
    cols: usize,
    rows: usize,
}

impl WinShape {
    fn new(win: usize) -> Self {
        Self {
            win,
            cols: win.next_multiple_of(LANES),
            rows: win.next_multiple_of(ROW_PAIR),
        }
    }

    /// Level padding that keeps every window read of a point that passes the
    /// in-image test inside the plane: `-win` on the left/top, `cols` (plus
    /// the bilinear neighbour) on the right, `rows` at the bottom.
    fn pad(&self) -> usize {
        self.cols.max(self.rows) + 1
    }

    fn chunks(&self) -> usize {
        self.cols / LANES
    }

    fn len(&self) -> usize {
        self.cols * self.rows
    }

    /// Tile index of window pixel `(x, y)`: rows go in pairs, and each
    /// [`LANES`]-wide chunk of a pair stores row `2p` then row `2p + 1`, so one
    /// 256-bit load holds both rows of a chunk.
    #[inline]
    fn tile(&self, x: usize, y: usize) -> usize {
        ((y / ROW_PAIR) * self.chunks() + x / LANES) * (ROW_PAIR * LANES)
            + (y % ROW_PAIR) * LANES
            + x % LANES
    }
}

impl LkPyramidU8 {
    /// Build the pyramid of `image` for tracking with window `win_size`.
    ///
    /// Levels follow OpenCV: each is `ceil(w / 2)` by `ceil(h / 2)` of the one
    /// below, made with the 5x5 Gaussian pyrDown, and the pyramid stops early
    /// at the level whose half size would be `<= win_size` in either axis.
    ///
    /// # Arguments
    ///
    /// * `image` - Grayscale frame.
    /// * `win_size` - LK window size in pixels (odd, >= 3).
    /// * `max_level` - Highest level (0 means the image only).
    /// * `with_derivatives` - Also compute the Scharr derivatives each level;
    ///   required for the previous frame.
    ///
    /// # Returns
    ///
    /// The pyramid.
    ///
    /// # Errors
    ///
    /// [`PyrLkU8Error::InvalidWindowSize`] or [`PyrLkU8Error::InvalidMaxLevel`].
    pub fn new(
        image: &Image<u8, 1>,
        win_size: usize,
        max_level: usize,
        with_derivatives: bool,
    ) -> Result<Self, PyrLkU8Error> {
        let mut pyramid = Self {
            levels: Vec::new(),
            win_size,
        };
        pyramid.rebuild(image, win_size, max_level, with_derivatives)?;
        Ok(pyramid)
    }

    /// Rebuild in place for a new frame, reusing the level buffers when the
    /// geometry is unchanged.
    ///
    /// # Arguments
    ///
    /// Same as [`LkPyramidU8::new`].
    ///
    /// # Returns
    ///
    /// `Ok(())` on success.
    ///
    /// # Errors
    ///
    /// Same as [`LkPyramidU8::new`].
    pub fn rebuild(
        &mut self,
        image: &Image<u8, 1>,
        win_size: usize,
        max_level: usize,
        with_derivatives: bool,
    ) -> Result<(), PyrLkU8Error> {
        if win_size < 3 || win_size.is_multiple_of(2) {
            return Err(PyrLkU8Error::InvalidWindowSize(win_size));
        }
        if max_level > PYR_LK_MAX_LEVEL {
            return Err(PyrLkU8Error::InvalidMaxLevel(max_level));
        }
        let pad = WinShape::new(win_size).pad();
        let sizes = level_sizes(image.size(), win_size, max_level);
        let reuse = self.win_size == win_size
            && self.levels.len() == sizes.len()
            && self
                .levels
                .iter()
                .zip(&sizes)
                .all(|(l, s)| l.image.width == s.width && l.image.height == s.height);
        if !reuse {
            self.levels = sizes
                .iter()
                .map(|s| Level {
                    image: Plane::new(s.width, s.height, pad),
                    deriv: None,
                })
                .collect();
        }
        self.win_size = win_size;
        for k in 0..self.levels.len() {
            let (below, rest) = self.levels.split_at_mut(k);
            let level = &mut rest[0];
            match below.last() {
                None => copy_into(image, &mut level.image),
                Some(prev) => downsample_into(&prev.image, &mut level.image),
            }
            level.image.reflect_border();
            if with_derivatives {
                let (w, h) = (level.image.width, level.image.height);
                let deriv = level
                    .deriv
                    .get_or_insert_with(|| (Plane::new(w, h, pad), Plane::new(w, h, pad)));
                scharr_deriv_i16(&level.image, &mut deriv.0, &mut deriv.1);
            } else {
                level.deriv = None;
            }
        }
        Ok(())
    }

    /// Number of levels (the effective `max_level + 1`).
    pub fn num_levels(&self) -> usize {
        self.levels.len()
    }

    /// Window size the levels are padded for.
    pub fn win_size(&self) -> usize {
        self.win_size
    }

    /// Size of level 0.
    pub fn size(&self) -> ImageSize {
        self.levels.first().map_or(
            ImageSize {
                width: 0,
                height: 0,
            },
            |l| ImageSize {
                width: l.image.width,
                height: l.image.height,
            },
        )
    }
}

/// Level sizes as `cv::buildOpticalFlowPyramid` makes them.
fn level_sizes(size: ImageSize, win: usize, max_level: usize) -> Vec<ImageSize> {
    let mut sizes = vec![size];
    let mut s = size;
    for _ in 0..max_level {
        s = ImageSize {
            width: s.width.div_ceil(2),
            height: s.height.div_ceil(2),
        };
        if s.width <= win || s.height <= win {
            break;
        }
        sizes.push(s);
    }
    sizes
}

fn copy_into(image: &Image<u8, 1>, plane: &mut Plane<u8>) {
    let (w, origin, stride) = (plane.width, plane.origin(), plane.stride);
    for (y, row) in image.as_slice().chunks_exact(w).enumerate() {
        plane.data[origin + y * stride..][..w].copy_from_slice(row);
    }
}

fn downsample_into(src: &Plane<u8>, dst: &mut Plane<u8>) {
    let (so, ss, sw, sh) = (src.origin(), src.stride, src.width, src.height);
    let (d_origin, d_stride, dw) = (dst.origin(), dst.stride, dst.width);
    pyrdown_c1_u8_strided(
        &src.data[so..],
        ss,
        sw,
        sh,
        &mut dst.data[d_origin..],
        d_stride,
        dw,
    );
}

/// OpenCV's `calcSharrDeriv`: unscaled 3x3 Scharr on a reflect-101 padded
/// level, as separate x and y planes whose padding stays zero.
fn scharr_deriv_i16(image: &Plane<u8>, ix: &mut Plane<i16>, iy: &mut Plane<i16>) {
    #[cfg(target_arch = "x86_64")]
    if crate::simd::cpu_features().has_avx2 {
        // SAFETY: AVX2 is present.
        unsafe { scharr_deriv_i16_avx2(image, ix, iy) };
        return;
    }
    scharr_deriv_i16_rows(image, ix, iy);
}

/// The portable row loops compiled for AVX2.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn scharr_deriv_i16_avx2(image: &Plane<u8>, ix: &mut Plane<i16>, iy: &mut Plane<i16>) {
    scharr_deriv_i16_rows(image, ix, iy);
}

#[inline(always)]
fn scharr_deriv_i16_rows(image: &Plane<u8>, ix: &mut Plane<i16>, iy: &mut Plane<i16>) {
    let w = image.width;
    let mut smooth = vec![0i16; w + 2];
    let mut diff = vec![0i16; w + 2];
    for y in 0..image.height as isize {
        let start = image.index(-1, y);
        let above = &image.data[start - image.stride..][..w + 2];
        let center = &image.data[start..][..w + 2];
        let below = &image.data[start + image.stride..][..w + 2];
        let it = smooth
            .iter_mut()
            .zip(diff.iter_mut())
            .zip(above)
            .zip(center)
            .zip(below);
        for ((((s, d), &a), &c), &b) in it {
            let (a, c, b) = (i16::from(a), i16::from(c), i16::from(b));
            *s = (a + b) * 3 + c * 10;
            *d = b - a;
        }
        let o = ix.index(0, y);
        let rows = ix.data[o..o + w]
            .iter_mut()
            .zip(&mut iy.data[o..o + w])
            .zip(smooth.windows(3))
            .zip(diff.windows(3));
        for (((gx, gy), s), d) in rows {
            *gx = s[2] - s[0];
            *gy = (d[0] + d[2]) * 3 + d[1] * 10;
        }
    }
}

/// Q14 bilinear weights `[w00, w01, w10, w11]` for fractions `(a, b)`, rounded as OpenCV does.
#[inline]
fn q14_weights(a: f32, b: f32) -> [i32; 4] {
    let one = W_ONE as f32;
    let w00 = ((1.0 - a) * (1.0 - b) * one).round_ties_even() as i32;
    let w01 = (a * (1.0 - b) * one).round_ties_even() as i32;
    let w10 = ((1.0 - a) * b * one).round_ties_even() as i32;
    [w00, w01, w10, W_ONE - w00 - w01 - w10]
}

#[inline]
fn descale(v: i32, n: i32) -> i32 {
    (v + (1 << (n - 1))) >> n
}

/// The previous frame's window at one level: samples (`I * 32`) and
/// derivatives in [`WinShape::tile`] layout, padding lanes zero.
struct Window {
    shape: WinShape,
    i: Vec<i16>,
    ix: Vec<i16>,
    iy: Vec<i16>,
}

impl Window {
    fn new(shape: WinShape) -> Self {
        Self {
            shape,
            i: vec![0; shape.len()],
            ix: vec![0; shape.len()],
            iy: vec![0; shape.len()],
        }
    }

    /// Samples the window at integer corner `(x0, y0)` with weights `w` and
    /// returns the integer structure tensor `(A11, A12, A22)`. Exact, so
    /// every ISA path fills the same window.
    fn extract(
        &mut self,
        level: &Level,
        deriv: &(Plane<i16>, Plane<i16>),
        x0: isize,
        y0: isize,
        w: [i32; 4],
    ) -> (i64, i64, i64) {
        #[cfg(target_arch = "x86_64")]
        {
            // SAFETY: SSE2 is part of x86_64; the corner passed the in-image
            // test, so with `WinShape::pad` every read lies inside the planes.
            return unsafe { self.extract_sse2(level, deriv, x0, y0, w) };
        }
        #[allow(unreachable_code)]
        self.extract_scalar(level, deriv, x0, y0, w)
    }

    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "sse2")]
    unsafe fn extract_sse2(
        &mut self,
        level: &Level,
        deriv: &(Plane<i16>, Plane<i16>),
        x0: isize,
        y0: isize,
        w: [i32; 4],
    ) -> (i64, i64, i64) {
        use std::arch::x86_64::*;
        let s = self.shape;
        let img = &level.image;
        let (dxp, dyp) = deriv;
        let (is, ds) = (img.stride, dxp.stride);
        debug_assert!(img.index(x0, y0 + s.win as isize) + s.cols < img.data.len());
        let qw0 = _mm_set1_epi32((w[1] << 16) | (w[0] & 0xffff));
        let qw1 = _mm_set1_epi32((w[3] << 16) | (w[2] & 0xffff));
        let half_i = _mm_set1_epi32(1 << (SAMPLE_SHIFT - 1));
        let half_d = _mm_set1_epi32(1 << (W_BITS - 1));
        let z = _mm_setzero_si128();
        // Lanes `x >= win` of the last chunk stay zero.
        let mut tail = [0i16; LANES];
        tail.iter_mut()
            .enumerate()
            .for_each(|(k, m)| *m = if (s.cols - LANES + k) < s.win { -1 } else { 0 });
        let tail = _mm_loadu_si128(tail.as_ptr() as *const __m128i);
        // Each lane takes two products of at most MAX_DERIV^2 per chunk and row.
        let flush =
            (i64::from(i32::MAX) / (2 * s.chunks() as i64 * MAX_DERIV * MAX_DERIV)).max(1) as usize;
        let (mut a11, mut a12, mut a22) = (0i64, 0i64, 0i64);
        let (mut q11, mut q12, mut q22) = (z, z, z);
        for y in 0..s.win {
            let pi = img.data.as_ptr().add(img.index(x0, y0 + y as isize));
            let px = dxp.data.as_ptr().add(dxp.index(x0, y0 + y as isize));
            let py = dyp.data.as_ptr().add(dyp.index(x0, y0 + y as isize));
            for c in 0..s.chunks() {
                let x = c * LANES;
                let ld8 = |p: *const u8| _mm_unpacklo_epi8(_mm_loadl_epi64(p as *const __m128i), z);
                let (a0, a1) = (ld8(pi.add(x)), ld8(pi.add(x + 1)));
                let (b0, b1) = (ld8(pi.add(x + is)), ld8(pi.add(x + is + 1)));
                let bil = |a0, a1, b0, b1, half| {
                    let lo = _mm_add_epi32(
                        _mm_madd_epi16(_mm_unpacklo_epi16(a0, a1), qw0),
                        _mm_madd_epi16(_mm_unpacklo_epi16(b0, b1), qw1),
                    );
                    let hi = _mm_add_epi32(
                        _mm_madd_epi16(_mm_unpackhi_epi16(a0, a1), qw0),
                        _mm_madd_epi16(_mm_unpackhi_epi16(b0, b1), qw1),
                    );
                    (_mm_add_epi32(lo, half), _mm_add_epi32(hi, half))
                };
                let (lo, hi) = bil(a0, a1, b0, b1, half_i);
                let iv = _mm_packs_epi32(
                    _mm_srai_epi32(lo, SAMPLE_SHIFT),
                    _mm_srai_epi32(hi, SAMPLE_SHIFT),
                );
                let ld16 = |p: *const i16| _mm_loadu_si128(p as *const __m128i);
                let (lo, hi) = bil(
                    ld16(px.add(x)),
                    ld16(px.add(x + 1)),
                    ld16(px.add(x + ds)),
                    ld16(px.add(x + ds + 1)),
                    half_d,
                );
                let gx = _mm_packs_epi32(_mm_srai_epi32(lo, W_BITS), _mm_srai_epi32(hi, W_BITS));
                let (lo, hi) = bil(
                    ld16(py.add(x)),
                    ld16(py.add(x + 1)),
                    ld16(py.add(x + ds)),
                    ld16(py.add(x + ds + 1)),
                    half_d,
                );
                let gy = _mm_packs_epi32(_mm_srai_epi32(lo, W_BITS), _mm_srai_epi32(hi, W_BITS));
                let (iv, gx, gy) = if c + 1 == s.chunks() {
                    (
                        _mm_and_si128(iv, tail),
                        _mm_and_si128(gx, tail),
                        _mm_and_si128(gy, tail),
                    )
                } else {
                    (iv, gx, gy)
                };
                let t = s.tile(x, y);
                _mm_storeu_si128(self.i.as_mut_ptr().add(t) as *mut __m128i, iv);
                _mm_storeu_si128(self.ix.as_mut_ptr().add(t) as *mut __m128i, gx);
                _mm_storeu_si128(self.iy.as_mut_ptr().add(t) as *mut __m128i, gy);
                q11 = _mm_add_epi32(q11, _mm_madd_epi16(gx, gx));
                q12 = _mm_add_epi32(q12, _mm_madd_epi16(gx, gy));
                q22 = _mm_add_epi32(q22, _mm_madd_epi16(gy, gy));
            }
            if (y + 1).is_multiple_of(flush) || y + 1 == s.win {
                a11 += hsum_epi32_sse2(q11);
                a12 += hsum_epi32_sse2(q12);
                a22 += hsum_epi32_sse2(q22);
                (q11, q12, q22) = (z, z, z);
            }
        }
        (a11, a12, a22)
    }

    fn extract_scalar(
        &mut self,
        level: &Level,
        deriv: &(Plane<i16>, Plane<i16>),
        x0: isize,
        y0: isize,
        w: [i32; 4],
    ) -> (i64, i64, i64) {
        let s = self.shape;
        let img = &level.image;
        let (dxp, dyp) = deriv;
        let (mut a11, mut a12, mut a22) = (0i64, 0i64, 0i64);
        for y in 0..s.win {
            let ri = img.index(x0, y0 + y as isize);
            let rd = dxp.index(x0, y0 + y as isize);
            let (is, ds) = (img.stride, dxp.stride);
            for x in 0..s.win {
                let p = |d: &[u8], o: usize| i32::from(d[o]);
                let q = |d: &[i16], o: usize| i32::from(d[o]);
                let (i0, d0) = (ri + x, rd + x);
                let ival = p(&img.data, i0) * w[0]
                    + p(&img.data, i0 + 1) * w[1]
                    + p(&img.data, i0 + is) * w[2]
                    + p(&img.data, i0 + is + 1) * w[3];
                let bil = |d: &[i16]| {
                    q(d, d0) * w[0]
                        + q(d, d0 + 1) * w[1]
                        + q(d, d0 + ds) * w[2]
                        + q(d, d0 + ds + 1) * w[3]
                };
                let ixv = descale(bil(&dxp.data), W_BITS);
                let iyv = descale(bil(&dyp.data), W_BITS);
                let t = s.tile(x, y);
                self.i[t] = descale(ival, SAMPLE_SHIFT) as i16;
                self.ix[t] = ixv as i16;
                self.iy[t] = iyv as i16;
                a11 += i64::from(ixv * ixv);
                a12 += i64::from(ixv * iyv);
                a22 += i64::from(iyv * iyv);
            }
        }
        (a11, a12, a22)
    }
}

/// Window rows per i32 flush so no lane overflows: each lane gathers two
/// products of at most `MAX_DIFF * MAX_DERIV` per chunk and row.
fn rows_per_flush(shape: WinShape) -> usize {
    let per_row = 2 * shape.chunks() as i64 * MAX_DIFF * MAX_DERIV;
    ((i64::from(i32::MAX) / per_row) as usize / ROW_PAIR * ROW_PAIR).max(ROW_PAIR)
}

/// Mismatch vector `(sum (J - I) Ix, sum (J - I) Iy)` of the window at
/// integer corner `j0` of `next` with weights `w`. Exact, so every ISA path
/// returns the same integers.
fn mismatch(next: &Plane<u8>, j0: usize, w: [i32; 4], win: &Window) -> (i64, i64) {
    #[cfg(target_arch = "x86_64")]
    {
        if crate::simd::cpu_features().has_avx2 {
            // SAFETY: AVX2 is present; `j0` passed the in-image test, so with
            // the level padding of `WinShape::pad` every read lies in `next`.
            return unsafe { mismatch_avx2(next, j0, w, win) };
        }
        // SAFETY: SSE2 is part of x86_64; same bounds as above.
        return unsafe { mismatch_sse2(next, j0, w, win) };
    }
    #[allow(unreachable_code)]
    mismatch_scalar(next, j0, w, win)
}

fn mismatch_scalar(next: &Plane<u8>, j0: usize, w: [i32; 4], win: &Window) -> (i64, i64) {
    let s = win.shape;
    let st = next.stride;
    let (mut b1, mut b2) = (0i64, 0i64);
    for y in 0..s.rows {
        for x in 0..s.cols {
            let o = j0 + y * st + x;
            let j = |o: usize| i32::from(next.data[o]);
            let jv = j(o) * w[0] + j(o + 1) * w[1] + j(o + st) * w[2] + j(o + st + 1) * w[3];
            let t = s.tile(x, y);
            let diff = descale(jv, SAMPLE_SHIFT) - i32::from(win.i[t]);
            b1 += i64::from(diff * i32::from(win.ix[t]));
            b2 += i64::from(diff * i32::from(win.iy[t]));
        }
    }
    (b1, b2)
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "sse2")]
unsafe fn mismatch_sse2(next: &Plane<u8>, j0: usize, w: [i32; 4], win: &Window) -> (i64, i64) {
    use std::arch::x86_64::*;
    let s = win.shape;
    let st = next.stride;
    debug_assert!(j0 + (s.rows) * st + s.cols < next.data.len());
    let jp = next.data.as_ptr().add(j0);
    let (ip, xp, yp) = (win.i.as_ptr(), win.ix.as_ptr(), win.iy.as_ptr());
    let qw0 = _mm_set1_epi32((w[1] << 16) | (w[0] & 0xffff));
    let qw1 = _mm_set1_epi32((w[3] << 16) | (w[2] & 0xffff));
    let half = _mm_set1_epi32(1 << (SAMPLE_SHIFT - 1));
    let z = _mm_setzero_si128();
    let flush = rows_per_flush(s);
    let (mut b1, mut b2) = (0i64, 0i64);
    let mut y = 0;
    while y < s.rows {
        let end = (y + flush).min(s.rows);
        let (mut q1, mut q2) = (z, z);
        for yy in y..end {
            let r0 = jp.add(yy * st);
            let r1 = r0.add(st);
            for c in 0..s.chunks() {
                let x = c * LANES;
                let ld = |p: *const u8| _mm_unpacklo_epi8(_mm_loadl_epi64(p as *const __m128i), z);
                let (a0, a1, c0, c1) = (
                    ld(r0.add(x)),
                    ld(r0.add(x + 1)),
                    ld(r1.add(x)),
                    ld(r1.add(x + 1)),
                );
                let lo = _mm_add_epi32(
                    _mm_madd_epi16(_mm_unpacklo_epi16(a0, a1), qw0),
                    _mm_madd_epi16(_mm_unpacklo_epi16(c0, c1), qw1),
                );
                let hi = _mm_add_epi32(
                    _mm_madd_epi16(_mm_unpackhi_epi16(a0, a1), qw0),
                    _mm_madd_epi16(_mm_unpackhi_epi16(c0, c1), qw1),
                );
                let lo = _mm_srai_epi32(_mm_add_epi32(lo, half), SAMPLE_SHIFT);
                let hi = _mm_srai_epi32(_mm_add_epi32(hi, half), SAMPLE_SHIFT);
                let t = s.tile(x, yy);
                let i = _mm_loadu_si128(ip.add(t) as *const __m128i);
                let diff = _mm_sub_epi16(_mm_packs_epi32(lo, hi), i);
                let gx = _mm_loadu_si128(xp.add(t) as *const __m128i);
                let gy = _mm_loadu_si128(yp.add(t) as *const __m128i);
                q1 = _mm_add_epi32(q1, _mm_madd_epi16(diff, gx));
                q2 = _mm_add_epi32(q2, _mm_madd_epi16(diff, gy));
            }
        }
        b1 += hsum_epi32_sse2(q1);
        b2 += hsum_epi32_sse2(q2);
        y = end;
    }
    (b1, b2)
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "sse2")]
unsafe fn hsum_epi32_sse2(v: std::arch::x86_64::__m128i) -> i64 {
    let mut lanes = [0i32; 4];
    std::arch::x86_64::_mm_storeu_si128(lanes.as_mut_ptr() as *mut _, v);
    lanes.iter().map(|&l| i64::from(l)).sum()
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn mismatch_avx2(next: &Plane<u8>, j0: usize, w: [i32; 4], win: &Window) -> (i64, i64) {
    use std::arch::x86_64::*;
    let s = win.shape;
    let st = next.stride;
    debug_assert!(j0 + (s.rows) * st + s.cols < next.data.len());
    let jp = next.data.as_ptr().add(j0);
    let (ip, xp, yp) = (win.i.as_ptr(), win.ix.as_ptr(), win.iy.as_ptr());
    let qw0 = _mm256_set1_epi32((w[1] << 16) | (w[0] & 0xffff));
    let qw1 = _mm256_set1_epi32((w[3] << 16) | (w[2] & 0xffff));
    let half = _mm256_set1_epi32(1 << (SAMPLE_SHIFT - 1));
    let flush = rows_per_flush(s);
    let (mut b1, mut b2) = (0i64, 0i64);
    // Rows `y` and `y + 1` of 8 u8 at `p`, widened to 16 u16 (row y in the low half).
    macro_rules! ld2 {
        ($p:expr) => {{
            let p: *const u8 = $p;
            let a = _mm_loadl_epi64(p as *const __m128i);
            let b = _mm_loadl_epi64(p.add(st) as *const __m128i);
            _mm256_cvtepu8_epi16(_mm_unpacklo_epi64(a, b))
        }};
    }
    let mut y = 0;
    while y < s.rows {
        let end = (y + flush).min(s.rows);
        let (mut q1, mut q2) = (_mm256_setzero_si256(), _mm256_setzero_si256());
        for yy in (y..end).step_by(ROW_PAIR) {
            let r0 = jp.add(yy * st);
            let r1 = r0.add(st);
            for c in 0..s.chunks() {
                let x = c * LANES;
                let (a0, a1, c0, c1) = (
                    ld2!(r0.add(x)),
                    ld2!(r0.add(x + 1)),
                    ld2!(r1.add(x)),
                    ld2!(r1.add(x + 1)),
                );
                let lo = _mm256_add_epi32(
                    _mm256_madd_epi16(_mm256_unpacklo_epi16(a0, a1), qw0),
                    _mm256_madd_epi16(_mm256_unpacklo_epi16(c0, c1), qw1),
                );
                let hi = _mm256_add_epi32(
                    _mm256_madd_epi16(_mm256_unpackhi_epi16(a0, a1), qw0),
                    _mm256_madd_epi16(_mm256_unpackhi_epi16(c0, c1), qw1),
                );
                let lo = _mm256_srai_epi32(_mm256_add_epi32(lo, half), SAMPLE_SHIFT);
                let hi = _mm256_srai_epi32(_mm256_add_epi32(hi, half), SAMPLE_SHIFT);
                let t = s.tile(x, yy);
                let i = _mm256_loadu_si256(ip.add(t) as *const __m256i);
                let diff = _mm256_sub_epi16(_mm256_packs_epi32(lo, hi), i);
                let gx = _mm256_loadu_si256(xp.add(t) as *const __m256i);
                let gy = _mm256_loadu_si256(yp.add(t) as *const __m256i);
                q1 = _mm256_add_epi32(q1, _mm256_madd_epi16(diff, gx));
                q2 = _mm256_add_epi32(q2, _mm256_madd_epi16(diff, gy));
            }
        }
        b1 += hsum_epi32_avx2(q1);
        b2 += hsum_epi32_avx2(q2);
        y = end;
    }
    (b1, b2)
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn hsum_epi32_avx2(v: std::arch::x86_64::__m256i) -> i64 {
    let mut lanes = [0i32; 8];
    std::arch::x86_64::_mm256_storeu_si256(lanes.as_mut_ptr() as *mut _, v);
    lanes.iter().map(|&l| i64::from(l)).sum()
}

/// `sum |J - I|` over the real window, in window units.
fn abs_error(next: &Plane<u8>, j0: usize, w: [i32; 4], win: &Window) -> i64 {
    let s = win.shape;
    let st = next.stride;
    let mut sum = 0i64;
    for y in 0..s.win {
        for x in 0..s.win {
            let o = j0 + y * st + x;
            let j = |o: usize| i32::from(next.data[o]);
            let jv = j(o) * w[0] + j(o + 1) * w[1] + j(o + st) * w[2] + j(o + st + 1) * w[3];
            let diff = descale(jv, SAMPLE_SHIFT) - i32::from(win.i[s.tile(x, y)]);
            sum += i64::from(diff.abs());
        }
    }
    sum
}

/// Integer corner of a window whose top-left is at `p`, if the window still
/// overlaps the level as OpenCV requires (`-win <= floor(p) < size`).
#[inline]
fn corner(p: [f32; 2], win: usize, width: usize, height: usize) -> Option<[isize; 2]> {
    let (x, y) = (p[0].floor(), p[1].floor());
    let w = win as f32;
    if x < -w || x >= width as f32 || y < -w || y >= height as f32 {
        return None;
    }
    Some([x as isize, y as isize])
}

struct TrackOut {
    pt: [f32; 2],
    status: u8,
    error: f32,
}

/// Inverse-compositional LK of one point through all levels, OpenCV's
/// `LKTrackerInvoker` step for step.
fn track_point(
    prev: &LkPyramidU8,
    next: &LkPyramidU8,
    p0: [f32; 2],
    guess: Option<[f32; 2]>,
    params: &PyrLKParams,
    win: &mut Window,
) -> TrackOut {
    let shape = win.shape;
    let half_win = (shape.win as f32 - 1.0) * 0.5;
    let max_level = params.max_level.min(prev.levels.len() - 1);
    let use_eps = matches!(params.term_criteria, TermCriteria::Eps | TermCriteria::Both);
    let eps2 = f64::from(params.epsilon) * f64::from(params.epsilon);
    let area = (shape.win * shape.win) as f32;
    let mut out = TrackOut {
        pt: [0.0; 2],
        status: 1,
        error: 0.0,
    };
    for level in (0..=max_level).rev() {
        let scale = 1.0 / (1u32 << level) as f32;
        let prev_pt = [p0[0] * scale, p0[1] * scale];
        out.pt = if level == max_level {
            guess.map_or(prev_pt, |g| [g[0] * scale, g[1] * scale])
        } else {
            [out.pt[0] * 2.0, out.pt[1] * 2.0]
        };
        let (lp, ln) = (&prev.levels[level], &next.levels[level]);
        let Some(deriv) = lp.deriv.as_ref() else {
            out.status = 0;
            return out;
        };
        let (lw, lh) = (lp.image.width, lp.image.height);
        let tl = [prev_pt[0] - half_win, prev_pt[1] - half_win];
        let Some([ix0, iy0]) = corner(tl, shape.win, lw, lh) else {
            if level == 0 {
                out.status = 0;
                out.error = 0.0;
            }
            continue;
        };
        let w = q14_weights(tl[0] - ix0 as f32, tl[1] - iy0 as f32);
        let (ia11, ia12, ia22) = win.extract(lp, deriv, ix0, iy0, w);
        let (a11, a12, a22) = (
            ia11 as f32 * FLT_SCALE,
            ia12 as f32 * FLT_SCALE,
            ia22 as f32 * FLT_SCALE,
        );
        let det = a11 * a22 - a12 * a12;
        let min_eig =
            (a22 + a11 - ((a11 - a22) * (a11 - a22) + 4.0 * a12 * a12).sqrt()) / (2.0 * area);
        if min_eig < params.min_eigen_threshold || det < f32::EPSILON {
            if level == 0 {
                out.status = 0;
            }
            continue;
        }
        let inv_det = 1.0 / det;
        let mut np = [out.pt[0] - half_win, out.pt[1] - half_win];
        let mut prev_delta = [0.0f32; 2];
        for it in 0..params.max_iter {
            let Some([jx, jy]) = corner(np, shape.win, lw, lh) else {
                if level == 0 {
                    out.status = 0;
                }
                break;
            };
            let w = q14_weights(np[0] - jx as f32, np[1] - jy as f32);
            let (ib1, ib2) = mismatch(&ln.image, ln.image.index(jx, jy), w, win);
            let (b1, b2) = (ib1 as f32 * FLT_SCALE, ib2 as f32 * FLT_SCALE);
            let delta = [
                (a12 * b2 - a22 * b1) * inv_det,
                (a12 * b1 - a11 * b2) * inv_det,
            ];
            np = [np[0] + delta[0], np[1] + delta[1]];
            out.pt = [np[0] + half_win, np[1] + half_win];
            let d2 = f64::from(delta[0]) * f64::from(delta[0])
                + f64::from(delta[1]) * f64::from(delta[1]);
            if use_eps && d2 <= eps2 {
                break;
            }
            if it > 0
                && (delta[0] + prev_delta[0]).abs() < OSCILLATION_TOL
                && (delta[1] + prev_delta[1]).abs() < OSCILLATION_TOL
            {
                out.pt = [out.pt[0] - delta[0] * 0.5, out.pt[1] - delta[1] * 0.5];
                break;
            }
            prev_delta = delta;
        }
        if out.status == 1 && level == 0 {
            let tl = [out.pt[0] - half_win, out.pt[1] - half_win];
            let Some([jx, jy]) = corner(tl, shape.win, lw, lh) else {
                out.status = 0;
                continue;
            };
            let w = q14_weights(tl[0] - jx as f32, tl[1] - jy as f32);
            let sum = abs_error(&ln.image, ln.image.index(jx, jy), w, win);
            out.error = sum as f32 / (SAMPLE_UNIT * area);
        }
    }
    out
}

/// Sparse pyramidal Lucas-Kanade on u8 pyramids with OpenCV's fixed-point
/// arithmetic and status rules.
///
/// This is `cv::calcOpticalFlowPyrLK` without the `OPTFLOW_LK_GET_MIN_EIGENVALS`
/// flag: Q14 bilinear weights, i16 Scharr derivatives, exact integer window
/// sums, the oscillation half-step, and status 0 when a window leaves the
/// image (by more than the window size), when the minimum eigenvalue of the
/// structure tensor divided by the window area is below
/// `params.min_eigen_threshold`, or when its determinant is below
/// `f32::EPSILON`. `params.border_mode` is ignored: borders are the
/// pyramid's reflect-101 padding. With [`TermCriteria::Count`] the epsilon
/// test is skipped; with `Eps` or `Both` iterations stop once
/// `|delta| <= params.epsilon`, and `params.max_iter` always caps them.
///
/// The error of a tracked point is the mean absolute difference between the
/// two windows in pixel values. Status-0 points keep their last estimate.
///
/// # Arguments
///
/// * `prev` - Pyramid of the previous frame, built with derivatives.
/// * `next` - Pyramid of the next frame, same geometry.
/// * `prev_pts` - Points to track, in level-0 pixels.
/// * `next_pts_in` - Initial guesses, used when `params.use_initial_flow`.
/// * `params` - LK parameters; `win_size` must equal the pyramids' window.
///
/// # Returns
///
/// The tracked points, status and error per point.
///
/// # Errors
///
/// [`PyrLkU8Error`] when the pyramids do not match each other or `params`,
/// the previous one lacks derivatives, or the initial flow is inconsistent.
///
/// # Example
///
/// ```
/// use kornia_image::{Image, ImageSize};
/// use kornia_imgproc::optical_flow_pyr_lk::{calc_optical_flow_pyr_lk_u8, LkPyramidU8, PyrLKParams};
///
/// let size = ImageSize { width: 64, height: 64 };
/// let data: Vec<u8> = (0..64 * 64).map(|i| ((i % 64) * (i / 64) % 251) as u8).collect();
/// let img = Image::<u8, 1>::new(size, data).unwrap();
/// let params = PyrLKParams { max_level: 1, ..PyrLKParams::default() };
/// let prev = LkPyramidU8::new(&img, params.win_size, params.max_level, true).unwrap();
/// let next = LkPyramidU8::new(&img, params.win_size, params.max_level, false).unwrap();
/// let res = calc_optical_flow_pyr_lk_u8(&prev, &next, &[[32.0, 32.0]], None, &params).unwrap();
/// assert_eq!(res.status, vec![1]);
/// assert!((res.next_pts[0][0] - 32.0).abs() < 1e-3);
/// ```
pub fn calc_optical_flow_pyr_lk_u8(
    prev: &LkPyramidU8,
    next: &LkPyramidU8,
    prev_pts: &[[f32; 2]],
    next_pts_in: Option<&[[f32; 2]]>,
    params: &PyrLKParams,
) -> Result<PyrLKResult, PyrLkU8Error> {
    check_inputs(prev, next, prev_pts, next_pts_in, params)?;
    let guesses = next_pts_in.filter(|_| params.use_initial_flow);
    let shape = WinShape::new(prev.win_size);
    let mut tracked: Vec<TrackOut> = Vec::with_capacity(prev_pts.len());
    prev_pts
        .par_chunks(POINTS_PER_TASK)
        .enumerate()
        .map(|(k, pts)| {
            let mut win = Window::new(shape);
            pts.iter()
                .enumerate()
                .map(|(i, &p)| {
                    let g = guesses.map(|g| g[k * POINTS_PER_TASK + i]);
                    track_point(prev, next, p, g, params, &mut win)
                })
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>()
        .into_iter()
        .for_each(|chunk| tracked.extend(chunk));
    Ok(PyrLKResult {
        next_pts: tracked.iter().map(|t| t.pt).collect(),
        status: tracked.iter().map(|t| t.status).collect(),
        error: tracked.iter().map(|t| t.error).collect(),
    })
}

fn check_inputs(
    prev: &LkPyramidU8,
    next: &LkPyramidU8,
    prev_pts: &[[f32; 2]],
    next_pts_in: Option<&[[f32; 2]]>,
    params: &PyrLKParams,
) -> Result<(), PyrLkU8Error> {
    let same = prev.win_size == next.win_size
        && prev.levels.len() == next.levels.len()
        && prev
            .levels
            .iter()
            .zip(&next.levels)
            .all(|(a, b)| a.image.width == b.image.width && a.image.height == b.image.height);
    if !same || prev.levels.is_empty() {
        return Err(PyrLkU8Error::PyramidMismatch {
            prev_size: prev.size(),
            prev_levels: prev.levels.len(),
            prev_win: prev.win_size,
            next_size: next.size(),
            next_levels: next.levels.len(),
            next_win: next.win_size,
        });
    }
    if params.win_size != prev.win_size {
        return Err(PyrLkU8Error::WindowMismatch {
            params: params.win_size,
            pyramid: prev.win_size,
        });
    }
    if prev.levels.iter().any(|l| l.deriv.is_none()) {
        return Err(PyrLkU8Error::MissingDerivatives);
    }
    if params.use_initial_flow {
        let Some(g) = next_pts_in else {
            return Err(PyrLkU8Error::InitialFlowMissing);
        };
        if g.len() != prev_pts.len() {
            return Err(PyrLkU8Error::InitialFlowLengthMismatch {
                expected: prev_pts.len(),
                provided: g.len(),
            });
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pyramid::pyrdown_u8;
    use std::path::PathBuf;

    fn data_dir() -> PathBuf {
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../tests/data")
    }

    fn frame(name: &str) -> Image<u8, 1> {
        let img = kornia_io::png::read_image_png_mono8(data_dir().join(name)).unwrap();
        Image::new(img.size(), img.as_slice().to_vec()).unwrap()
    }

    /// `tests/data/frontend_opencv.bin`, written by `benches/opencv_frontend_fixtures.py`.
    struct Fixture {
        pts: Vec<[f32; 2]>,
        cv_next: Vec<[f32; 2]>,
        cv_status: Vec<u8>,
        crop: ImageSize,
        cv_pyrdown: Vec<u8>,
    }

    fn fixture() -> Fixture {
        let b = std::fs::read(data_dir().join("frontend_opencv.bin")).unwrap();
        let u32_at = |o: usize| u32::from_le_bytes(b[o..o + 4].try_into().unwrap()) as usize;
        let f32_at = |o: usize| f32::from_le_bytes(b[o..o + 4].try_into().unwrap());
        let n = u32_at(0);
        let pts_at = |o: usize| {
            (0..n)
                .map(|i| [f32_at(o + 8 * i), f32_at(o + 8 * i + 4)])
                .collect()
        };
        let (p0, p1, ps) = (4, 4 + 8 * n, 4 + 16 * n);
        let pc = ps + n;
        let crop = ImageSize {
            width: u32_at(pc),
            height: u32_at(pc + 4),
        };
        Fixture {
            pts: pts_at(p0),
            cv_next: pts_at(p1),
            cv_status: b[ps..ps + n].to_vec(),
            crop,
            cv_pyrdown: b[pc + 8..].to_vec(),
        }
    }

    fn track(prev: &Image<u8, 1>, next: &Image<u8, 1>, pts: &[[f32; 2]]) -> PyrLKResult {
        let params = PyrLKParams::default();
        let p = LkPyramidU8::new(prev, params.win_size, params.max_level, true).unwrap();
        let q = LkPyramidU8::new(next, params.win_size, params.max_level, false).unwrap();
        calc_optical_flow_pyr_lk_u8(&p, &q, pts, None, &params).unwrap()
    }

    #[test]
    fn pyrdown_u8_matches_opencv_bytes() {
        let fx = fixture();
        let f1 = frame("mh01_frame1.png");
        let (w, h) = (fx.crop.width, fx.crop.height);
        let crop: Vec<u8> = f1
            .as_slice()
            .chunks_exact(f1.width())
            .take(h)
            .flat_map(|r| r[..w].to_vec())
            .collect();
        let src = Image::<u8, 1>::new(fx.crop, crop).unwrap();
        let half = ImageSize {
            width: w.div_ceil(2),
            height: h.div_ceil(2),
        };
        let mut dst = Image::<u8, 1>::from_size_val(half, 0).unwrap();
        pyrdown_u8(&src, &mut dst).unwrap();
        assert_eq!(dst.as_slice(), &fx.cv_pyrdown[..]);
    }

    /// Tolerances: status agrees on >= 98 % of the points and, where both
    /// track, the end points agree to 0.01 px at the median and 0.1 px at
    /// the worst. The arithmetic is OpenCV's except the tensor sums (exact
    /// integers here, f32 in OpenCV), so small float-order differences remain.
    #[test]
    fn lk_u8_matches_opencv() {
        let fx = fixture();
        let res = track(
            &frame("mh01_frame1.png"),
            &frame("mh01_frame2.png"),
            &fx.pts,
        );
        let n = fx.pts.len();
        let agree = res
            .status
            .iter()
            .zip(&fx.cv_status)
            .filter(|(a, b)| a == b)
            .count();
        assert!(agree * 100 >= n * 98, "status agree {agree}/{n}");
        let mut epe: Vec<f32> = (0..n)
            .filter(|&i| res.status[i] == 1 && fx.cv_status[i] == 1)
            .map(|i| {
                let (a, b) = (res.next_pts[i], fx.cv_next[i]);
                ((a[0] - b[0]).powi(2) + (a[1] - b[1]).powi(2)).sqrt()
            })
            .collect();
        epe.sort_by(f32::total_cmp);
        let (median, max) = (epe[epe.len() / 2], epe[epe.len() - 1]);
        assert!(
            median <= 0.01 && max <= 0.1,
            "EPE median {median} max {max}"
        );
    }

    #[test]
    fn lk_u8_drops_points_outside_the_image() {
        let f1 = frame("mh01_frame1.png");
        let res = track(&f1, &f1, &[[-40.0, 100.0], [100.0, 600.0], [300.0, 200.0]]);
        assert_eq!(res.status, vec![0, 0, 1]);
    }

    #[test]
    fn lk_u8_recovers_a_shift() {
        let size = ImageSize {
            width: 160,
            height: 120,
        };
        let pattern = |x: usize, y: usize| -> u8 {
            let (fx, fy) = (x as f32, y as f32);
            (128.0
                + 60.0 * (fx * 0.21).sin() * (fy * 0.17).cos()
                + 40.0 * (fx * 0.05 + fy * 0.09).sin()) as u8
        };
        let img = |dx: usize| {
            let d = (0..size.width * size.height)
                .map(|i| pattern(i % size.width + dx, i / size.width))
                .collect();
            Image::<u8, 1>::new(size, d).unwrap()
        };
        let pts = [[80.0, 60.0], [50.0, 40.0]];
        let res = track(&img(0), &img(3), &pts);
        for ((p, s), p0) in res.next_pts.iter().zip(&res.status).zip(&pts) {
            assert_eq!(*s, 1);
            assert!((p[1] - p0[1]).abs() < 0.05, "{p:?}");
        }
        assert!(
            (res.next_pts[0][0] - 77.0).abs() < 0.05,
            "{:?}",
            res.next_pts
        );
        assert!(
            (res.next_pts[1][0] - 47.0).abs() < 0.05,
            "{:?}",
            res.next_pts
        );
    }

    #[test]
    fn simd_mismatch_equals_scalar() {
        let f1 = frame("mh01_frame1.png");
        let f2 = frame("mh01_frame2.png");
        let shape = WinShape::new(21);
        let p = LkPyramidU8::new(&f1, 21, 0, true).unwrap();
        let q = LkPyramidU8::new(&f2, 21, 0, false).unwrap();
        let (lp, ln) = (&p.levels[0], &q.levels[0]);
        let Some(deriv) = lp.deriv.as_ref() else {
            panic!("derivatives missing");
        };
        let mut win = Window::new(shape);
        for &(x, y, fx, fy) in &[
            (-21isize, -21isize, 0.3f32, 0.7f32),
            (100, 200, 0.0, 0.0),
            (751, 479, 0.99, 0.5),
            (400, -5, 0.5, 0.25),
        ] {
            let w = q14_weights(fx, fy);
            let tensor = win.extract_scalar(lp, deriv, x, y, w);
            let mut simd = Window::new(shape);
            assert_eq!(simd.extract(lp, deriv, x, y, w), tensor);
            assert_eq!((&simd.i, &simd.ix, &simd.iy), (&win.i, &win.ix, &win.iy));
            let j0 = ln.image.index(x, y);
            let scalar = mismatch_scalar(&ln.image, j0, w, &win);
            assert_eq!(mismatch(&ln.image, j0, w, &win), scalar);
            #[cfg(target_arch = "x86_64")]
            {
                // SAFETY: SSE2 is baseline on x86_64; the corner is in range.
                assert_eq!(unsafe { mismatch_sse2(&ln.image, j0, w, &win) }, scalar);
                if crate::simd::cpu_features().has_avx2 {
                    // SAFETY: AVX2 checked above.
                    assert_eq!(unsafe { mismatch_avx2(&ln.image, j0, w, &win) }, scalar);
                }
            }
        }
    }

    #[test]
    fn lk_u8_rejects_bad_inputs() {
        let f1 = frame("mh01_frame1.png");
        let params = PyrLKParams::default();
        assert!(matches!(
            LkPyramidU8::new(&f1, 20, 3, true),
            Err(PyrLkU8Error::InvalidWindowSize(20))
        ));
        let p = LkPyramidU8::new(&f1, 21, 3, true).unwrap();
        let q = LkPyramidU8::new(&f1, 21, 3, false).unwrap();
        let q2 = LkPyramidU8::new(&f1, 21, 2, false).unwrap();
        assert!(matches!(
            calc_optical_flow_pyr_lk_u8(&q, &p, &[[1.0, 1.0]], None, &params),
            Err(PyrLkU8Error::MissingDerivatives)
        ));
        assert!(matches!(
            calc_optical_flow_pyr_lk_u8(&p, &q2, &[[1.0, 1.0]], None, &params),
            Err(PyrLkU8Error::PyramidMismatch { .. })
        ));
        let init = PyrLKParams {
            use_initial_flow: true,
            ..PyrLKParams::default()
        };
        assert!(matches!(
            calc_optical_flow_pyr_lk_u8(&p, &q, &[[1.0, 1.0]], None, &init),
            Err(PyrLkU8Error::InitialFlowMissing)
        ));
    }
}
