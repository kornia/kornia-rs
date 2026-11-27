//! Shi-Tomasi corners (the minimum-eigenvalue response) on `wide` vectors,
//! detected only in grid cells no existing point occupies.
//!
//! Reference: OpenCV `goodFeaturesToTrack` with a 3×3 Sobel and a 3×3 block
//! (`blockSize 3`, `useHarrisDetector false`): a quality threshold relative to
//! the strongest response, a 3×3 maximum (OpenCV's dilate-and-compare), a
//! minimum spacing on OpenCV's grid and a border mask. Departures, for a
//! tracker that tops up its points each frame: only cells no existing point
//! occupies are searched, so the quality threshold is relative to the
//! strongest response in those cells (and the one-pixel ring around each run
//! of them that the 3×3 maximum reads) rather than the whole image; the
//! spacing also holds against the existing points; and the
//! response can run on a 2×2 box-averaged image (`half_resolution`). The
//! tensor sums are exact integers held in f32, so the response does not
//! depend on the ISA.

use kornia_image::Image;
use wide::{f32x8, i16x8};

use crate::simd::lanes::{lane_starts, load16, load8_i16, load_i32, store_i32, LANES};

/// A response reads 2 pixels each side (Sobel, then the block), and the
/// 3×3 maximum one response more.
const SUPPORT: usize = 3;
/// End of a cell's point list.
const NIL: u32 = u32::MAX;

/// What [`good_features_to_track`] keeps.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct GoodFeaturesParams {
    /// Most corners one call appends.
    pub max_corners: usize,
    /// Fraction of the strongest response a corner must reach; `0` or more.
    pub quality: f32,
    /// No corner closer than this to another, or to an occupied point
    /// (pixels); finite and `0` or more.
    pub min_spacing_px: f32,
    /// Corners within this many pixels of the image edge are dropped.
    pub border_px: usize,
    /// Respond on a 2× downsampled image; corners are returned in full-size pixels.
    pub half_resolution: bool,
}

/// Buffers [`good_features_to_track`] reuses across calls.
#[derive(Debug, Clone, Default)]
pub struct GoodFeaturesScratch {
    /// The 2×2 box-averaged image under `half_resolution`.
    half: Vec<u8>,
    /// One value per source pixel, valid inside the empty cells.
    response: Vec<f32>,
    occupied_cells: Vec<bool>,
    candidates: Vec<(f32, [f32; 2])>,
    /// `Ix²`, `Ix Iy`, `Iy²` of one run of cells, and one row of their
    /// vertical block sums.
    products: [Vec<i32>; 3],
    sums: [Vec<i32>; 3],
    /// Spacing grid: per cell the newest point, per point the next one.
    cell_head: Vec<u32>,
    point_next: Vec<u32>,
    points: Vec<[f32; 2]>,
}

/// Appends to `out` the strongest Shi-Tomasi corners of `image` away from `occupied`.
///
/// Corners are searched only in the grid cells (of `min_spacing_px`) that no
/// occupied point falls in, kept when their response exceeds `quality` times
/// the strongest response there (including the one-pixel ring around each run
/// of empty cells, which the 3×3 maximum reads) and is a 3×3 maximum, and appended strongest
/// first (ties by row, then column) while they stay `min_spacing_px` from
/// every occupied and appended point. Allocation-free once `scratch` and
/// `out` have grown to the image size and corner count.
///
/// # Arguments
///
/// * `image` - Grayscale image.
/// * `occupied` - Existing points (full-size pixels); their cells are skipped.
/// * `params` - Detection parameters.
/// * `scratch` - Buffers reused across calls.
/// * `out` - Receives the corners (full-size pixels); existing entries are kept.
///
/// # Returns
///
/// Nothing; the corners are appended to `out`. An image smaller than 7 pixels
/// (at the detection resolution) or without texture appends none, and so do
/// invalid parameters: a negative or NaN `quality` (which would pass flat
/// pixels or none), or a negative or non-finite `min_spacing_px`.
///
/// # Errors
///
/// This function is infallible.
///
/// # Example
///
/// ```rust
/// use kornia_image::{Image, ImageSize};
/// use kornia_imgproc::features::{good_features_to_track, GoodFeaturesParams, GoodFeaturesScratch};
///
/// # fn main() -> Result<(), Box<dyn std::error::Error>> {
/// // A bright square: its four corners are the features.
/// let size = ImageSize { width: 64, height: 64 };
/// let data = (0..64 * 64)
///     .map(|i| if (20..44).contains(&(i % 64)) && (20..44).contains(&(i / 64)) { 220 } else { 30 })
///     .collect();
/// let image = Image::<u8, 1>::new(size, data)?;
/// let params = GoodFeaturesParams {
///     max_corners: 10,
///     quality: 0.05,
///     min_spacing_px: 8.0,
///     border_px: 4,
///     half_resolution: false,
/// };
/// let mut corners = Vec::new();
/// good_features_to_track(&image, &[], &params, &mut GoodFeaturesScratch::default(), &mut corners);
/// assert_eq!(corners.len(), 4);
/// # Ok(())
/// # }
/// ```
pub fn good_features_to_track(
    image: &Image<u8, 1>,
    occupied: &[[f32; 2]],
    params: &GoodFeaturesParams,
    scratch: &mut GoodFeaturesScratch,
    out: &mut Vec<[f32; 2]>,
) {
    let GoodFeaturesScratch {
        half,
        response,
        occupied_cells,
        candidates,
        products,
        sums,
        cell_head,
        point_next,
        points,
    } = scratch;
    let valid_quality = params.quality >= 0.0;
    let valid_spacing = params.min_spacing_px.is_finite() && params.min_spacing_px >= 0.0;
    if !(valid_quality && valid_spacing) {
        return;
    }
    let (w, h) = (image.width(), image.height());
    let (factor, offset) = if params.half_resolution {
        (2, 0.5)
    } else {
        (1, 0.0)
    };
    let source = Source {
        width: w / factor,
        height: h / factor,
        factor,
        offset,
    };
    if params.max_corners == 0 || source.width < 2 * SUPPORT + 1 || source.height < 2 * SUPPORT + 1
    {
        return;
    }
    let src: &[u8] = if params.half_resolution {
        half_image(image.as_slice(), w, &source, half);
        &half[..]
    } else {
        image.as_slice()
    };
    let cell = ((params.min_spacing_px / factor as f32).ceil() as usize).max(1);
    let cells = Cells {
        size: cell,
        cols: source.width.div_ceil(cell),
        rows: source.height.div_ceil(cell),
    };
    mark_occupied(occupied, &source, &cells, occupied_cells);
    response.resize(source.width * source.height, 0.0);
    let mut max_response = 0.0f32;
    for_each_region(&source, &cells, occupied_cells, |region| {
        let r = respond(src, &source, region.grown(), products, sums, response);
        max_response = max_response.max(r);
    });
    if max_response <= 0.0 {
        return;
    }
    let threshold = max_response * params.quality;
    let border = params.border_px as f32;
    let inside = |p: [f32; 2]| {
        (border..=(w as f32 - 1.0) - border).contains(&p[0])
            && (border..=(h as f32 - 1.0) - border).contains(&p[1])
    };
    candidates.clear();
    for_each_region(&source, &cells, occupied_cells, |region| {
        for y in region.y0..region.y1 {
            for x in region.x0..region.x1 {
                let r = response[y * source.width + x];
                if r > threshold && is_local_max(response, source.width, x, y) {
                    let p = source.full(x, y);
                    if inside(p) {
                        candidates.push((r, p));
                    }
                }
            }
        }
    });
    candidates.sort_unstable_by(|a, b| {
        b.0.total_cmp(&a.0)
            .then(a.1[1].total_cmp(&b.1[1]))
            .then(a.1[0].total_cmp(&b.1[0]))
    });
    let mut spacing = Spacing::new(params.min_spacing_px, w, h, cell_head, point_next, points);
    for &p in occupied {
        spacing.insert(p);
    }
    let mut added = 0;
    for &(_, p) in candidates.iter() {
        if spacing.is_clear(p) {
            spacing.insert(p);
            out.push(p);
            added += 1;
            if added == params.max_corners {
                break;
            }
        }
    }
}

/// The image the response runs on, and its map to full-size pixels.
struct Source {
    width: usize,
    height: usize,
    factor: usize,
    /// Full-size position of source pixel 0: the centre of its 2×2 block.
    offset: f32,
}

impl Source {
    fn full(&self, x: usize, y: usize) -> [f32; 2] {
        let f = self.factor as f32;
        [x as f32 * f + self.offset, y as f32 * f + self.offset]
    }

    fn cell_of(&self, p: [f32; 2], cells: &Cells) -> Option<usize> {
        let f = self.factor as f32;
        let (x, y) = ((p[0] - self.offset) / f, (p[1] - self.offset) / f);
        let inside =
            (0.0..self.width as f32).contains(&x) && (0.0..self.height as f32).contains(&y);
        if !inside {
            return None;
        }
        Some((y as usize / cells.size) * cells.cols + x as usize / cells.size)
    }
}

/// Detection cells, in source pixels.
struct Cells {
    size: usize,
    cols: usize,
    rows: usize,
}

/// Candidate pixels `[x0, x1) × [y0, y1)`.
#[derive(Clone, Copy)]
struct Region {
    x0: usize,
    x1: usize,
    y0: usize,
    y1: usize,
}

impl Region {
    /// The pixels whose responses the 3×3 maximum of this region reads.
    fn grown(self) -> Self {
        Self {
            x0: self.x0 - 1,
            x1: self.x1 + 1,
            y0: self.y0 - 1,
            y1: self.y1 + 1,
        }
    }
}

fn mark_occupied(occupied: &[[f32; 2]], source: &Source, cells: &Cells, marks: &mut Vec<bool>) {
    marks.clear();
    marks.resize(cells.cols * cells.rows, false);
    for &p in occupied {
        if let Some(c) = source.cell_of(p, cells) {
            marks[c] = true;
        }
    }
}

/// Calls `f` with every run of horizontally adjacent empty cells, clipped to
/// the pixels whose response and maximum stay inside the image. Runs, not
/// cells, so the vector rows are long.
fn for_each_region(source: &Source, cells: &Cells, marks: &[bool], mut f: impl FnMut(Region)) {
    let (lo, x_hi, y_hi) = (SUPPORT, source.width - SUPPORT, source.height - SUPPORT);
    for cy in 0..cells.rows {
        let (y0, y1) = ((cy * cells.size).max(lo), ((cy + 1) * cells.size).min(y_hi));
        if y0 >= y1 {
            continue;
        }
        let row = &marks[cy * cells.cols..(cy + 1) * cells.cols];
        let mut cx = 0;
        while cx < cells.cols {
            if row[cx] {
                cx += 1;
                continue;
            }
            let start = cx;
            while cx < cells.cols && !row[cx] {
                cx += 1;
            }
            let (x0, x1) = ((start * cells.size).max(lo), (cx * cells.size).min(x_hi));
            if x0 < x1 {
                f(Region { x0, x1, y0, y1 });
            }
        }
    }
}

/// Minimum-eigenvalue responses of `region` into `response`; returns their
/// maximum. The caller keeps `region` two pixels inside the image.
fn respond(
    src: &[u8],
    source: &Source,
    region: Region,
    products: &mut [Vec<i32>; 3],
    sums: &mut [Vec<i32>; 3],
    response: &mut [f32],
) -> f32 {
    let sw = source.width;
    // Products cover the region plus one pixel each side, for the block.
    let (pw, ph) = (region.x1 - region.x0 + 2, region.y1 - region.y0 + 2);
    for p in products.iter_mut() {
        p.resize(pw * ph, 0);
    }
    for pr in 0..ph {
        let o = (region.y0 - 1 + pr) * sw + region.x0 - 1;
        let row = pr * pw;
        if pw < LANES {
            (0..pw).for_each(|pc| gradient_products_one(src, sw, o + pc, products, row + pc));
        } else {
            for pc in lane_starts(pw) {
                gradient_products_lanes(src, sw, o + pc, products, row + pc);
            }
        }
    }
    for s in sums.iter_mut() {
        s.resize(pw, 0);
    }
    let rw = region.x1 - region.x0;
    let mut max = 0.0f32;
    for rr in 0..ph - 2 {
        for (s, p) in sums.iter_mut().zip(products.iter()) {
            let (r0, r1, r2) = (&p[rr * pw..], &p[(rr + 1) * pw..], &p[(rr + 2) * pw..]);
            if pw < LANES {
                (0..pw).for_each(|c| s[c] = r0[c] + r1[c] + r2[c]);
            } else {
                for c in lane_starts(pw) {
                    let v = load_i32(&r0[c..]) + load_i32(&r1[c..]) + load_i32(&r2[c..]);
                    store_i32(&mut s[c..], v);
                }
            }
        }
        let out = &mut response[(region.y0 + rr) * sw + region.x0..][..rw];
        if rw < LANES {
            for (x, o) in out.iter_mut().enumerate() {
                let [a, b, c] = sums.each_ref().map(|s| (s[x] + s[x + 1] + s[x + 2]) as f32);
                *o = min_eigen(a, b, c);
                max = max.max(*o);
            }
            continue;
        }
        for x in lane_starts(rw) {
            let [a, b, c] = sums.each_ref().map(|s| {
                (load_i32(&s[x..]) + load_i32(&s[x + 1..]) + load_i32(&s[x + 2..])).round_float()
            });
            let r = min_eigen_lanes(a, b, c).to_array();
            out[x..x + LANES].copy_from_slice(&r);
            max = r.iter().fold(max, |m, &v| m.max(v));
        }
    }
    max
}

/// 3×3 Sobel `(Ix, Iy)` at source offset `o`; products stored at `p`.
// `mul_widen` is `widening_mul` from wide 1.6; the old name keeps the crate's
// `wide = "1.4"` requirement honest.
#[allow(deprecated)]
fn gradient_products_lanes(
    src: &[u8],
    sw: usize,
    o: usize,
    products: &mut [Vec<i32>; 3],
    p: usize,
) {
    let l = |k: usize| load8_i16(&src[k..]);
    let (up, dn) = (o - sw, o + sw);
    let ix = (l(up + 1) + l(o + 1) * 2i16 + l(dn + 1)) - (l(up - 1) + l(o - 1) * 2i16 + l(dn - 1));
    let iy = (l(dn - 1) + l(dn) * 2i16 + l(dn + 1)) - (l(up - 1) + l(up) * 2i16 + l(up + 1));
    store_i32(&mut products[0][p..], ix.mul_widen(ix));
    store_i32(&mut products[1][p..], ix.mul_widen(iy));
    store_i32(&mut products[2][p..], iy.mul_widen(iy));
}

fn gradient_products_one(src: &[u8], sw: usize, o: usize, products: &mut [Vec<i32>; 3], p: usize) {
    let l = |k: usize| i32::from(src[k]);
    let (up, dn) = (o - sw, o + sw);
    let ix = (l(up + 1) + 2 * l(o + 1) + l(dn + 1)) - (l(up - 1) + 2 * l(o - 1) + l(dn - 1));
    let iy = (l(dn - 1) + 2 * l(dn) + l(dn + 1)) - (l(up - 1) + 2 * l(up) + l(up + 1));
    products[0][p] = ix * ix;
    products[1][p] = ix * iy;
    products[2][p] = iy * iy;
}

/// Smaller eigenvalue of `[[a, b], [b, c]]`.
#[inline]
fn min_eigen(a: f32, b: f32, c: f32) -> f32 {
    let (mean, half_diff) = ((a + c) * 0.5, (a - c) * 0.5);
    mean - (half_diff * half_diff + b * b).sqrt()
}

#[inline(always)]
fn min_eigen_lanes(a: f32x8, b: f32x8, c: f32x8) -> f32x8 {
    let (mean, half_diff) = ((a + c) * 0.5f32, (a - c) * 0.5f32);
    mean - (half_diff * half_diff + b * b).sqrt()
}

/// At least every 8-neighbour (OpenCV's dilate-and-compare).
fn is_local_max(response: &[f32], width: usize, x: usize, y: usize) -> bool {
    let r = response[y * width + x];
    (y - 1..=y + 1).all(|ny| (x - 1..=x + 1).all(|nx| response[ny * width + nx] <= r))
}

/// 2×2 box average, `(sum + 2) >> 2`.
fn half_image(src: &[u8], w: usize, source: &Source, half: &mut Vec<u8>) {
    let (sw, sh) = (source.width, source.height);
    half.resize(sw * sh, 0);
    let ones = i16x8::splat(1);
    for y in 0..sh {
        let (r0, r1) = (&src[2 * y * w..], &src[(2 * y + 1) * w..]);
        let row = &mut half[y * sw..(y + 1) * sw];
        if sw < LANES {
            for (x, o) in row.iter_mut().enumerate() {
                let s = |r: &[u8]| u16::from(r[2 * x]) + u16::from(r[2 * x + 1]);
                *o = ((s(r0) + s(r1) + 2) >> 2) as u8;
            }
            continue;
        }
        for x in lane_starts(sw) {
            let (a, b) = (load16(&r0[2 * x..]), load16(&r1[2 * x..]));
            let lo = i16x8::from_u8x16_low(a) + i16x8::from_u8x16_low(b);
            let hi = i16x8::from_u8x16_high(a) + i16x8::from_u8x16_high(b);
            let lo = ((lo.dot(ones) + 2i32) >> 2i32).to_array();
            let hi = ((hi.dot(ones) + 2i32) >> 2i32).to_array();
            for (o, v) in row[x..x + LANES].iter_mut().zip(lo.into_iter().chain(hi)) {
                *o = v as u8;
            }
        }
    }
}

/// Points kept so far, bucketed by `min_spacing_px` cells (OpenCV's grid).
struct Spacing<'a> {
    spacing: f32,
    cols: usize,
    rows: usize,
    head: &'a mut Vec<u32>,
    next: &'a mut Vec<u32>,
    points: &'a mut Vec<[f32; 2]>,
}

impl<'a> Spacing<'a> {
    fn new(
        spacing: f32,
        w: usize,
        h: usize,
        head: &'a mut Vec<u32>,
        next: &'a mut Vec<u32>,
        points: &'a mut Vec<[f32; 2]>,
    ) -> Self {
        let spacing = if spacing.is_finite() && spacing > 0.0 {
            spacing
        } else {
            0.0
        };
        let (cols, rows) = if spacing > 0.0 {
            (
                ((w as f32 / spacing).ceil() as usize).max(1),
                ((h as f32 / spacing).ceil() as usize).max(1),
            )
        } else {
            (0, 0)
        };
        head.clear();
        head.resize(cols * rows, NIL);
        next.clear();
        points.clear();
        Self {
            spacing,
            cols,
            rows,
            head,
            next,
            points,
        }
    }

    fn cell(&self, p: [f32; 2]) -> (usize, usize) {
        let at = |v: f32, n: usize| ((v / self.spacing).floor().max(0.0) as usize).min(n - 1);
        (at(p[0], self.cols), at(p[1], self.rows))
    }

    fn insert(&mut self, p: [f32; 2]) {
        if self.spacing <= 0.0 {
            return;
        }
        let (cx, cy) = self.cell(p);
        let c = cy * self.cols + cx;
        self.next.push(self.head[c]);
        self.head[c] = self.points.len() as u32;
        self.points.push(p);
    }

    /// No kept point lies closer than the spacing.
    fn is_clear(&self, p: [f32; 2]) -> bool {
        if self.spacing <= 0.0 {
            return true;
        }
        let (cx, cy) = self.cell(p);
        let s2 = self.spacing * self.spacing;
        for ny in cy.saturating_sub(1)..=(cy + 1).min(self.rows - 1) {
            for nx in cx.saturating_sub(1)..=(cx + 1).min(self.cols - 1) {
                let mut k = self.head[ny * self.cols + nx];
                while k != NIL {
                    let q = self.points[k as usize];
                    let (dx, dy) = (p[0] - q[0], p[1] - q[1]);
                    if dx * dx + dy * dy < s2 {
                        return false;
                    }
                    k = self.next[k as usize];
                }
            }
        }
        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use kornia_image::ImageSize;

    fn image(width: usize, height: usize, f: impl Fn(usize, usize) -> u8) -> Image<u8, 1> {
        let data = (0..width * height)
            .map(|i| f(i % width, i / width))
            .collect();
        Image::new(ImageSize { width, height }, data).unwrap()
    }

    /// Bright squares on a dark background; their corners are the features.
    fn squares() -> Image<u8, 1> {
        image(160, 120, |x, y| {
            let inside =
                |x0: usize, y0: usize| (x0..x0 + 24).contains(&x) && (y0..y0 + 24).contains(&y);
            if inside(20, 20) || inside(90, 30) || inside(40, 75) {
                220
            } else {
                30
            }
        })
    }

    fn params(half_resolution: bool) -> GoodFeaturesParams {
        GoodFeaturesParams {
            max_corners: 100,
            quality: 0.05,
            min_spacing_px: 10.0,
            border_px: 9,
            half_resolution,
        }
    }

    fn detect(img: &Image<u8, 1>, occupied: &[[f32; 2]], p: &GoodFeaturesParams) -> Vec<[f32; 2]> {
        let mut out = Vec::new();
        good_features_to_track(
            img,
            occupied,
            p,
            &mut GoodFeaturesScratch::default(),
            &mut out,
        );
        out
    }

    fn square_corners() -> Vec<[f32; 2]> {
        [(20.0, 20.0), (90.0, 30.0), (40.0, 75.0)]
            .iter()
            .flat_map(|&(x, y)| [[x, y], [x + 23.0, y], [x, y + 23.0], [x + 23.0, y + 23.0]])
            .collect()
    }

    fn near(p: [f32; 2], q: [f32; 2], tol: f32) -> bool {
        (p[0] - q[0]).abs() <= tol && (p[1] - q[1]).abs() <= tol
    }

    #[test]
    fn response_matches_a_scalar_reference() {
        let img = image(67, 45, |x, y| {
            ((x * 37 + y * 101 + (x * y) % 23) % 251) as u8
        });
        let mut scratch = GoodFeaturesScratch::default();
        good_features_to_track(&img, &[], &params(false), &mut scratch, &mut Vec::new());
        let (w, h) = (img.width(), img.height());
        let px = |x: usize, y: usize| i32::from(img.as_slice()[y * w + x]);
        let grad = |x: usize, y: usize| {
            let ix = (px(x + 1, y - 1) + 2 * px(x + 1, y) + px(x + 1, y + 1))
                - (px(x - 1, y - 1) + 2 * px(x - 1, y) + px(x - 1, y + 1));
            let iy = (px(x - 1, y + 1) + 2 * px(x, y + 1) + px(x + 1, y + 1))
                - (px(x - 1, y - 1) + 2 * px(x, y - 1) + px(x + 1, y - 1));
            (ix, iy)
        };
        for y in SUPPORT..h - SUPPORT {
            for x in SUPPORT..w - SUPPORT {
                let (mut a, mut b, mut c) = (0, 0, 0);
                for (gx, gy) in
                    (y - 1..=y + 1).flat_map(|v| (x - 1..=x + 1).map(move |u| grad(u, v)))
                {
                    (a, b, c) = (a + gx * gx, b + gx * gy, c + gy * gy);
                }
                // The smaller root of l^2 - (a + c) l + (a c - b^2), in f64,
                // against f32 rounding at the trace's magnitude.
                let (a, b, c) = (f64::from(a), f64::from(b), f64::from(c));
                let want = 0.5 * ((a + c) - ((a - c).powi(2) + 4.0 * b * b).sqrt());
                let got = f64::from(scratch.response[y * w + x]);
                let tolerance = 8.0 * f64::from(f32::EPSILON) * (a + c).max(1.0);
                assert!(
                    (got - want).abs() <= tolerance,
                    "({x}, {y}): {got} vs {want}"
                );
            }
        }
    }

    #[test]
    fn finds_square_corners_apart_and_inside_the_border() {
        let p = params(false);
        let found = detect(&squares(), &[], &p);
        for c in square_corners() {
            assert!(
                found.iter().any(|&f| near(f, c, 1.5)),
                "corner {c:?} missing from {found:?}"
            );
        }
        for (i, a) in found.iter().enumerate() {
            assert!(
                (9.0..=150.0).contains(&a[0]) && (9.0..=110.0).contains(&a[1]),
                "{a:?}"
            );
            for b in &found[i + 1..] {
                let d2 = (a[0] - b[0]).powi(2) + (a[1] - b[1]).powi(2);
                assert!(d2 >= 100.0, "{a:?} and {b:?} closer than the spacing");
            }
        }
        assert_eq!(found, detect(&squares(), &[], &p), "deterministic");
    }

    #[test]
    fn half_resolution_returns_full_size_pixels() {
        let found = detect(&squares(), &[], &params(true));
        for c in square_corners() {
            assert!(
                found.iter().any(|&f| near(f, c, 2.5)),
                "corner {c:?} missing from {found:?}"
            );
        }
    }

    #[test]
    fn skips_occupied_cells_and_keeps_the_spacing_to_them() {
        let occupied = [[20.0, 20.0], [113.0, 30.0]];
        let found = detect(&squares(), &occupied, &params(false));
        for o in occupied {
            assert!(
                found
                    .iter()
                    .all(|&f| (f[0] - o[0]).powi(2) + (f[1] - o[1]).powi(2) >= 100.0),
                "{o:?} refilled: {found:?}"
            );
        }
        assert!(found.iter().any(|&f| near(f, [43.0, 43.0], 1.5)));
    }

    #[test]
    fn appends_at_most_max_corners() {
        let p = GoodFeaturesParams {
            max_corners: 3,
            ..params(false)
        };
        let mut out = vec![[-1.0, -1.0]];
        good_features_to_track(
            &squares(),
            &[],
            &p,
            &mut GoodFeaturesScratch::default(),
            &mut out,
        );
        assert_eq!(out.len(), 4);
        assert_eq!(out[0], [-1.0, -1.0]);
    }

    #[test]
    fn invalid_parameters_append_no_corners() {
        for p in [
            GoodFeaturesParams {
                quality: f32::NAN,
                ..params(false)
            },
            GoodFeaturesParams {
                quality: -0.1,
                ..params(false)
            },
            GoodFeaturesParams {
                min_spacing_px: f32::INFINITY,
                ..params(false)
            },
            GoodFeaturesParams {
                min_spacing_px: -1.0,
                ..params(false)
            },
        ] {
            assert!(detect(&squares(), &[], &p).is_empty(), "{p:?}");
        }
        assert!(!detect(&squares(), &[], &params(false)).is_empty());
    }

    /// The half-resolution path is the full-resolution detector on the 2×2
    /// box average (rounded half up), at half the spacing, with each corner
    /// mapped to the centre of its 2×2 block: `2x + 0.5`.
    #[test]
    fn half_resolution_is_full_resolution_on_the_box_average() {
        use crate::test_fixtures as cv;
        let frame = cv::kernels().frame1;
        let (w, h) = (frame.width(), frame.height());
        let (hw, hh) = (w / 2, h / 2);
        let src = frame.as_slice();
        let box_average = (0..hw * hh)
            .map(|i| {
                let (x, y) = (2 * (i % hw), 2 * (i / hw));
                let s = [(x, y), (x + 1, y), (x, y + 1), (x + 1, y + 1)]
                    .iter()
                    .map(|&(u, v)| u32::from(src[v * w + u]))
                    .sum::<u32>();
                ((s + 2) >> 2) as u8
            })
            .collect();
        let half = Image::new(
            ImageSize {
                width: hw,
                height: hh,
            },
            box_average,
        )
        .unwrap();
        let full = GoodFeaturesParams {
            max_corners: 300,
            quality: 0.01,
            min_spacing_px: 12.0,
            border_px: 0,
            half_resolution: true,
        };
        let reference = GoodFeaturesParams {
            min_spacing_px: full.min_spacing_px / 2.0,
            half_resolution: false,
            ..full
        };
        let ours = detect(&frame, &[], &full);
        let want: Vec<[f32; 2]> = detect(&half, &[], &reference)
            .into_iter()
            .map(|p| [2.0 * p[0] + 0.5, 2.0 * p[1] + 0.5])
            .collect();
        assert!(want.len() > 50, "{} corners", want.len());
        assert_eq!(ours, want);
    }

    #[test]
    fn a_flat_or_tiny_image_has_no_corners() {
        assert!(detect(&image(64, 48, |_, _| 77), &[], &params(false)).is_empty());
        assert!(detect(&image(12, 12, |x, _| (x * 20) as u8), &[], &params(true)).is_empty());
    }

    #[test]
    fn matches_opencv_good_features_to_track() {
        use crate::test_fixtures as cv;
        let fx = cv::kernels();
        let p = GoodFeaturesParams {
            max_corners: cv::MAX_CORNERS,
            quality: cv::QUALITY,
            min_spacing_px: cv::MIN_DISTANCE,
            border_px: cv::BORDER,
            half_resolution: false,
        };
        let ours = detect(&fx.frame1, &[], &p);
        // OpenCV takes its quality threshold over the whole image, borders
        // included, and breaks response ties in its own order.
        let matched = fx
            .corners
            .iter()
            .filter(|&&c| ours.iter().any(|&o| near(o, c, 0.5)))
            .count();
        assert!(
            matched * 10 >= fx.corners.len() * 9,
            "{matched} of {} OpenCV corners",
            fx.corners.len()
        );
        assert!(
            ours.len().abs_diff(fx.corners.len()) * 10 <= fx.corners.len(),
            "{} vs {}",
            ours.len(),
            fx.corners.len()
        );
    }
}
