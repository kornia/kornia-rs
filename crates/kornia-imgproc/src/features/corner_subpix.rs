//! Sub-pixel corner refinement.
//!
//! Reference: OpenCV `cornerSubPix`: each corner moves to where the image
//! gradients in its window are orthogonal to the offsets from it. The same
//! Gaussian window weights, bilinear patch with replicated borders
//! (`getRectSubPix`), iteration cap and convergence rules. Departures: no zero
//! zone, `half_win` is capped at [`CORNER_SUB_PIX_MAX_HALF_WIN`], and the window sums run on
//! `wide` f32 lanes and are added up in f64, where OpenCV keeps every term in
//! f64, so results agree to rounding, not bit for bit.

use kornia_image::Image;
use wide::{f32x8, i32x8};

use crate::simd::lanes::{lane_starts, load8_i16, load_f32, LANES};

/// Largest `half_win` of [`corner_sub_pix`]; larger values are clamped to it,
/// so the buffers stay on the stack.
pub const CORNER_SUB_PIX_MAX_HALF_WIN: usize = 15;
const MAX_COLS: usize = (2 * CORNER_SUB_PIX_MAX_HALF_WIN + 1).next_multiple_of(LANES);
/// Patch rows carry one more column each side for the central differences.
const MAX_STRIDE: usize = MAX_COLS + 2;
const MAX_ROWS: usize = 2 * CORNER_SUB_PIX_MAX_HALF_WIN + 3;
/// OpenCV's iteration cap.
const MAX_ITERS: usize = 100;

/// Refines `corners` in place to sub-pixel accuracy.
///
/// Each corner iterates over a `(2 half_win + 1)²` window for at most
/// `max_iter` steps or until a step is below `epsilon` pixels. A corner that
/// moves more than `half_win` in either axis keeps its input value, as
/// OpenCV's. Nothing moves when `half_win` is 0, the image is smaller than
/// `2 half_win + 5` in either axis, or the corner is outside it. No
/// allocation.
///
/// # Arguments
///
/// * `image` - Grayscale image.
/// * `corners` - Corner positions (pixels), refined in place.
/// * `half_win` - Half the window side, clamped to [`CORNER_SUB_PIX_MAX_HALF_WIN`].
/// * `max_iter` - Iteration cap per corner, clamped to `1..=100`.
/// * `epsilon` - Stop when a step moves less than this (pixels).
///
/// # Returns
///
/// Nothing; `corners` holds the refined positions.
///
/// # Errors
///
/// This function is infallible.
///
/// # Example
///
/// ```rust
/// use kornia_image::{Image, ImageSize};
/// use kornia_imgproc::features::corner_sub_pix;
///
/// # fn main() -> Result<(), Box<dyn std::error::Error>> {
/// // A blurred square whose top-left corner sits at (30.3, 22.6).
/// let edge = |v: f32| 1.0 / (1.0 + (-1.6 * v).exp());
/// let data = (0..64 * 48)
///     .map(|i| {
///         let (x, y) = ((i % 64) as f32, (i / 64) as f32);
///         (30.0 + 190.0 * edge(x - 30.3) * edge(y - 22.6)).round() as u8
///     })
///     .collect();
/// let image = Image::<u8, 1>::new(ImageSize { width: 64, height: 48 }, data)?;
/// let mut corners = [[31.0, 22.0]];
/// corner_sub_pix(&image, &mut corners, 5, 30, 0.01);
/// assert!((corners[0][0] - 30.3).abs() < 0.3 && (corners[0][1] - 22.6).abs() < 0.3);
/// # Ok(())
/// # }
/// ```
pub fn corner_sub_pix(
    image: &Image<u8, 1>,
    corners: &mut [[f32; 2]],
    half_win: usize,
    max_iter: usize,
    epsilon: f32,
) {
    let half_win = half_win.min(CORNER_SUB_PIX_MAX_HALF_WIN);
    let (w, h) = (image.width(), image.height());
    if half_win == 0 || w < 2 * half_win + 5 || h < 2 * half_win + 5 {
        return;
    }
    let window = Window::new(half_win);
    let max_iters = max_iter.clamp(1, MAX_ITERS);
    let eps2 = f64::from(epsilon.max(0.0)).powi(2);
    let mut patch = [0f32; MAX_ROWS * MAX_STRIDE];
    for corner in corners.iter_mut() {
        *corner = refine(image, *corner, &window, &mut patch, max_iters, eps2);
    }
}

/// The window's weights and offsets, per column (rows use the same ones).
struct Window {
    half: usize,
    /// `2 half + 1`.
    size: usize,
    /// `size` rounded up to whole vectors; the extra lanes weigh zero.
    cols: usize,
    stride: usize,
    /// OpenCV's mask factor `exp(-x²)`, `x = (j - half) / half`.
    weights: [f32; MAX_COLS],
    /// `j - half`.
    offsets: [f32; MAX_COLS],
}

impl Window {
    fn new(half: usize) -> Self {
        let size = 2 * half + 1;
        let cols = size.next_multiple_of(LANES);
        let mut weights = [0f32; MAX_COLS];
        let mut offsets = [0f32; MAX_COLS];
        for (j, (weight, offset)) in weights.iter_mut().zip(&mut offsets).enumerate().take(size) {
            let x = (j as f32 - half as f32) / half as f32;
            *weight = (-x * x).exp();
            *offset = j as f32 - half as f32;
        }
        Self {
            half,
            size,
            cols,
            stride: cols + 2,
            weights,
            offsets,
        }
    }
}

fn refine(
    image: &Image<u8, 1>,
    start: [f32; 2],
    window: &Window,
    patch: &mut [f32],
    max_iters: usize,
    eps2: f64,
) -> [f32; 2] {
    let (w, h) = (image.width() as f32, image.height() as f32);
    let inside = |c: [f32; 2]| (0.0..w).contains(&c[0]) && (0.0..h).contains(&c[1]);
    if !inside(start) {
        return start;
    }
    let mut c = start;
    for _ in 0..max_iters {
        fill_patch(image, c, window, patch);
        let [a, b, cc, bb1, bb2] = window_sums(patch, window);
        let det = a * cc - b * b;
        if det.abs() <= f64::EPSILON * f64::EPSILON {
            break;
        }
        let scale = 1.0 / det;
        let next = [
            (f64::from(c[0]) + cc * scale * bb1 - b * scale * bb2) as f32,
            (f64::from(c[1]) - b * scale * bb1 + a * scale * bb2) as f32,
        ];
        let (dx, dy) = (next[0] - c[0], next[1] - c[1]);
        let err = f64::from(dx * dx + dy * dy);
        c = next;
        if !inside(c) || err <= eps2 {
            break;
        }
    }
    let half = window.half as f32;
    if (c[0] - start[0]).abs() > half || (c[1] - start[1]).abs() > half {
        return start;
    }
    c
}

/// `(size + 2)²` bilinear samples centred on `centre`, rows `stride` apart;
/// outside the image the nearest pixel is used (OpenCV's `getRectSubPix`).
fn fill_patch(image: &Image<u8, 1>, centre: [f32; 2], window: &Window, patch: &mut [f32]) {
    let (w, h) = (image.width(), image.height());
    let data = image.as_slice();
    let reach = (window.half + 1) as f32;
    let tl = [centre[0] - reach, centre[1] - reach];
    let (fx, fy) = (tl[0].floor(), tl[1].floor());
    let (a, b) = (tl[0] - fx, tl[1] - fy);
    let k = [(1.0 - a) * (1.0 - b), a * (1.0 - b), (1.0 - a) * b, a * b];
    let (x0, y0) = (fx as isize, fy as isize);
    let rows = window.size + 2;
    let stride = window.stride;
    let interior = x0 >= 0 && y0 >= 0 && x0 as usize + stride < w && y0 as usize + rows < h;
    if interior {
        let kv = k.map(f32x8::splat);
        for r in 0..rows {
            let o = (y0 as usize + r) * w + x0 as usize;
            for c in lane_starts(stride) {
                let p = [o + c, o + c + 1, o + c + w, o + c + w + 1]
                    .map(|i| i32x8::from_i16x8(load8_i16(&data[i..])).round_float());
                let v = p[0] * kv[0] + p[1] * kv[1] + p[2] * kv[2] + p[3] * kv[3];
                patch[r * stride + c..][..LANES].copy_from_slice(v.as_array());
            }
        }
        return;
    }
    let px = |x: isize, y: isize| {
        let (x, y) = (x.clamp(0, w as isize - 1), y.clamp(0, h as isize - 1));
        f32::from(data[y as usize * w + x as usize])
    };
    for r in 0..rows {
        for c in 0..stride {
            let (x, y) = (x0 + c as isize, y0 + r as isize);
            patch[r * stride + c] = px(x, y) * k[0]
                + px(x + 1, y) * k[1]
                + px(x, y + 1) * k[2]
                + px(x + 1, y + 1) * k[3];
        }
    }
}

/// OpenCV's `a`, `b`, `c` (the weighted gradient tensor) and `bb1`, `bb2`
/// (its product with the offsets) over the window.
fn window_sums(patch: &[f32], window: &Window) -> [f64; 5] {
    let stride = window.stride;
    let zero = f32x8::splat(0.0);
    let mut acc = [zero; 5];
    let rows = window.offsets.iter().zip(&window.weights).take(window.size);
    for (i, (&py, &wy)) in rows.enumerate() {
        let py = f32x8::splat(py);
        let (up, mid, down) = (i * stride, (i + 1) * stride, (i + 2) * stride);
        for j in (0..window.cols).step_by(LANES) {
            let tgx = load_f32(&patch[mid + j + 2..]) - load_f32(&patch[mid + j..]);
            let tgy = load_f32(&patch[down + j + 1..]) - load_f32(&patch[up + j + 1..]);
            let m = load_f32(&window.weights[j..]) * wy;
            let px = load_f32(&window.offsets[j..]);
            let (gxx, gxy, gyy) = (tgx * tgx * m, tgx * tgy * m, tgy * tgy * m);
            acc[0] += gxx;
            acc[1] += gxy;
            acc[2] += gyy;
            acc[3] += gxx * px + gxy * py;
            acc[4] += gxy * px + gyy * py;
        }
    }
    acc.map(|v| v.to_array().iter().map(|&l| f64::from(l)).sum())
}

#[cfg(test)]
mod tests {
    use super::*;
    use kornia_image::ImageSize;

    /// A blurred bright square whose top-left corner sits at `(cx, cy)`.
    fn blurred_square(cx: f32, cy: f32) -> Image<u8, 1> {
        let (width, height) = (64, 48);
        let edge = |v: f32| 1.0 / (1.0 + (-1.6 * v).exp());
        let data = (0..width * height)
            .map(|i| {
                let (x, y) = ((i % width) as f32, (i / width) as f32);
                (30.0 + 190.0 * edge(x - cx) * edge(y - cy)).round() as u8
            })
            .collect();
        Image::new(ImageSize { width, height }, data).unwrap()
    }

    #[test]
    fn moves_a_corner_onto_a_subpixel_corner() {
        let img = blurred_square(30.3, 22.6);
        let mut corners = [[31.0, 22.0], [29.6, 23.4]];
        corner_sub_pix(&img, &mut corners, 5, 30, 0.01);
        for c in corners {
            // The blurred corner pulls the estimate slightly along the diagonal.
            assert!(
                (c[0] - 30.3).abs() < 0.3 && (c[1] - 22.6).abs() < 0.3,
                "{c:?}"
            );
        }
    }

    #[test]
    fn interior_and_border_patches_agree() {
        let img = blurred_square(30.3, 22.6);
        let (w, h, pad) = (img.width(), img.height(), 40);
        // `img` with its border replicated `pad` pixels out: a centre near
        // `img`'s corner takes the clamped path there and the interior path here.
        let size = ImageSize {
            width: w + 2 * pad,
            height: h + 2 * pad,
        };
        let padded = (0..size.width * size.height)
            .map(|i| {
                let x = (i % size.width).saturating_sub(pad).min(w - 1);
                let y = (i / size.width).saturating_sub(pad).min(h - 1);
                img.as_slice()[y * w + x]
            })
            .collect();
        let big = Image::new(size, padded).unwrap();
        let window = Window::new(5);
        let mut border = [0f32; MAX_ROWS * MAX_STRIDE];
        let mut inner = [0f32; MAX_ROWS * MAX_STRIDE];
        fill_patch(&img, [3.4, 2.7], &window, &mut border);
        fill_patch(
            &big,
            [3.4 + pad as f32, 2.7 + pad as f32],
            &window,
            &mut inner,
        );
        let n = (window.size + 2) * window.stride;
        for (i, (a, b)) in border[..n].iter().zip(&inner[..n]).enumerate() {
            assert!((a - b).abs() < 1e-3, "{i}: {a} vs {b}");
        }
    }

    #[test]
    fn keeps_corners_on_a_flat_or_tiny_image() {
        let flat = Image::new(
            ImageSize {
                width: 32,
                height: 32,
            },
            vec![90; 32 * 32],
        )
        .unwrap();
        let mut corners = [[16.25, 15.75]];
        corner_sub_pix(&flat, &mut corners, 4, 20, 0.01);
        assert_eq!(corners, [[16.25, 15.75]]);
        // A textured corner the same window does refine on a larger image, so
        // only the size guard keeps it still on the 8x8 one (2·2 + 5 = 9 > 8).
        let corner = |size: usize| {
            let edge = |v: f32| 1.0 / (1.0 + (-1.6 * v).exp());
            let data = (0..size * size)
                .map(|i| {
                    let (x, y) = ((i % size) as f32, (i / size) as f32);
                    (30.0 + 190.0 * edge(x - 4.3) * edge(y - 3.6)).round() as u8
                })
                .collect();
            Image::new(
                ImageSize {
                    width: size,
                    height: size,
                },
                data,
            )
            .unwrap()
        };
        let mut large = [[4.0, 4.0]];
        corner_sub_pix(&corner(32), &mut large, 2, 20, 0.01);
        assert_ne!(large, [[4.0, 4.0]], "the texture must be refinable");
        let mut tiny = [[4.0, 4.0]];
        corner_sub_pix(&corner(8), &mut tiny, 2, 20, 0.01);
        assert_eq!(tiny, [[4.0, 4.0]]);
    }

    #[test]
    fn matches_opencv_corner_sub_pix() {
        use crate::test_fixtures as cv;
        let fx = cv::kernels();
        let mut ours = fx.corners.clone();
        corner_sub_pix(
            &fx.frame1,
            &mut ours,
            cv::SUBPIX_HALF_WIN,
            cv::SUBPIX_MAX_ITER,
            cv::EPSILON,
        );
        // f32 lane sums against OpenCV's f64 ones, and its getRectSubPix rounding.
        let close = ours
            .iter()
            .zip(&fx.refined)
            .filter(|(o, c)| (o[0] - c[0]).abs() <= 0.02 && (o[1] - c[1]).abs() <= 0.02)
            .count();
        assert!(
            close * 100 >= ours.len() * 95,
            "{close} of {} within 0.02 px",
            ours.len()
        );
    }
}
