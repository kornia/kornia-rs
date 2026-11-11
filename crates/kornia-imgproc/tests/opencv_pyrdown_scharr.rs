//! Single-channel `pyrdown_f32` and `scharr_spatial_gradient_float` against
//! OpenCV, at sizes whose interiors run the vector kernels and their scalar
//! tails (752x480, 751x479) and at small odd sizes (17x9, 33x34).
//!
//! The input is an integer plane in `0..251`, regenerated here by
//! [`synthetic_plane`], so no image fixture is stored. Every weight of both
//! kernels is a dyadic fraction (pyrdown `/256`, Scharr `/32`), so with integer
//! input every product and partial sum is exact in `f32`, in kornia and in
//! OpenCV alike, whatever the summation order. The tolerance is therefore zero:
//! the pins below are compared with `==`.
//!
//! Each output is pinned by its sum, a position-weighted sum
//! `sum(v[i] * (i % 1009 + 1))` (both exact in `f64`), and eight sampled pixels
//! covering corners, borders and the interior. Generated with
//! opencv-python-headless 5.0.0:
//!
//! ```python
//! import cv2, numpy as np
//! def plane(w, h):
//!     y, x = np.mgrid[0:h, 0:w].astype(np.int64)
//!     return ((x * x + 3 * y * y + 7 * x * y + 5 * x + 11 * y) % 251).astype(np.float32)
//! def pins(a):
//!     h, w = a.shape
//!     f = a.astype(np.float64).ravel()
//!     wsum = float((f * (np.arange(f.size) % 1009 + 1)).sum())
//!     pts = [(0, 0), (1, 1), (w // 2, h // 2), (w - 1, h - 1), (w - 2, max(h - 2, 0)),
//!            (min(9, w - 1), min(1, h - 1)), (min(40, w - 1), min(3, h - 1)), (w - 1, 0)]
//!     s = ", ".join(f"({x}, {y}, {float(a[y, x])!r})" for x, y in pts)
//!     return f"{float(f.sum())!r}, {wsum!r}, [{s}]"
//! for w, h in [(17, 9), (33, 34), (751, 479), (752, 480)]:
//!     src = plane(w, h)
//!     p = cv2.pyrDown(src)
//!     gx = cv2.Scharr(src, cv2.CV_32F, 1, 0, scale=1 / 32, borderType=cv2.BORDER_REPLICATE)
//!     gy = cv2.Scharr(src, cv2.CV_32F, 0, 1, scale=1 / 32, borderType=cv2.BORDER_REPLICATE)
//!     print(f"PYR (({w}, {h}), {pins(p)}),")
//!     print(f"SCX (({w}, {h}), {pins(gx)}),")
//!     print(f"SCY (({w}, {h}), {pins(gy)}),")
//! ```

use kornia_image::{Image, ImageError, ImageSize};
use kornia_imgproc::filter::scharr_spatial_gradient_float;
use kornia_imgproc::pyramid::pyrdown_f32;

/// `((src_width, src_height), sum, weighted_sum, [(x, y, value); 8])` of one
/// OpenCV output.
type Pins = ((usize, usize), f64, f64, [(usize, usize, f32); 8]);

// The f32 pins are exact dyadic values, written in full as numpy prints them;
// clippy would shorten them to the nearest-f32 form, which reads less exactly.
#[allow(clippy::excessive_precision)]
#[rustfmt::skip]
const PYRDOWN: [Pins; 4] = [
    ((17, 9), 5184.37890625, 122898.3828125, [(0, 0, 19.9375), (1, 1, 80.0), (4, 2, 120.0546875), (8, 4, 78.703125), (7, 3, 111.33984375), (8, 1, 134.2265625), (8, 3, 103.140625), (8, 0, 122.109375)]),
    ((33, 34), 34994.54296875, 5036602.11328125, [(0, 0, 19.9375), (1, 1, 80.0), (8, 8, 123.80859375), (16, 16, 141.1875), (15, 15, 136.140625), (9, 1, 106.89453125), (16, 3, 144.125), (16, 0, 120.234375)]),
    ((751, 479), 11285021.6171875, 5681205514.964844, [(0, 0, 19.9375), (1, 1, 80.0), (188, 120, 121.453125), (375, 239, 113.984375), (374, 238, 118.546875), (9, 1, 106.89453125), (40, 3, 117.59375), (375, 0, 226.546875)]),
    ((752, 480), 11284085.984375, 5681108477.2578125, [(0, 0, 19.9375), (1, 1, 80.0), (188, 120, 121.453125), (375, 239, 123.96484375), (374, 238, 118.546875), (9, 1, 106.89453125), (40, 3, 117.59375), (375, 0, 226.9921875)]),
];

#[rustfmt::skip]
const SCHARR_DX: [Pins; 4] = [
    ((17, 9), 279.0, -11550.25, [(0, 0, 3.65625), (1, 1, 14.0), (8, 4, 25.46875), (16, 8, -56.625), (15, 7, -17.96875), (9, 1, 6.46875), (16, 3, 28.5), (16, 0, 18.65625)]),
    ((33, 34), 511.0, -52050.3125, [(0, 0, 3.65625), (1, 1, 14.0), (16, 17, 30.5), (32, 33, 23.34375), (31, 32, 16.46875), (9, 1, 6.46875), (32, 3, 44.5), (32, 0, 34.65625)]),
    ((751, 479), 3127.0, 2207312.84375, [(0, 0, 3.65625), (1, 1, 14.0), (375, 239, -58.46875), (750, 478, 39.84375), (749, 477, -28.96875), (9, 1, 6.46875), (40, 3, 4.03125), (750, 0, -0.34375)]),
    ((752, 480), -1402.0, 647835.5, [(0, 0, 3.65625), (1, 1, 14.0), (376, 240, -49.46875), (751, 479, 20.8125), (750, 478, -19.96875), (9, 1, 6.46875), (40, 3, 4.03125), (751, 0, 0.65625)]),
];

#[rustfmt::skip]
const SCHARR_DY: [Pins; 4] = [
    ((17, 9), 579.0, 1601.75, [(0, 0, 7.65625), (1, 1, 24.0), (8, 4, -10.96875), (16, 8, -18.625), (15, 7, 8.96875), (9, 1, -21.96875), (16, 3, 14.1875), (16, 0, 62.34375)]),
    ((33, 34), -206.0, 26915.6875, [(0, 0, 7.65625), (1, 1, 24.0), (16, 17, -26.0), (32, 33, -36.65625), (31, 32, 19.96875), (9, 1, -21.96875), (32, 3, 0.6875), (32, 0, -7.15625)]),
    ((751, 479), 333.0, 5149955.90625, [(0, 0, 7.65625), (1, 1, 24.0), (375, 239, 30.46875), (750, 478, 46.34375), (749, 477, -17.96875), (9, 1, -21.96875), (40, 3, -43.96875), (750, 0, -4.15625)]),
    ((752, 480), -2248.0, -10007137.375, [(0, 0, 7.65625), (1, 1, 24.0), (376, 240, 43.46875), (751, 479, 29.3125), (750, 478, -4.96875), (9, 1, -21.96875), (40, 3, -43.96875), (751, 0, -0.65625)]),
];

fn synthetic_plane(size: ImageSize) -> Result<Image<f32, 1>, ImageError> {
    let data = (0..size.height)
        .flat_map(|y| {
            (0..size.width)
                .map(move |x| ((x * x + 3 * y * y + 7 * x * y + 5 * x + 11 * y) % 251) as f32)
        })
        .collect();
    Image::new(size, data)
}

fn assert_pins(out: &Image<f32, 1>, pins: &Pins, what: &str) {
    let ((w, h), sum, weighted_sum, samples) = pins;
    let data = out.as_slice();
    let got_sum: f64 = data.iter().map(|&v| v as f64).sum();
    let got_weighted: f64 = data
        .iter()
        .enumerate()
        .map(|(i, &v)| v as f64 * ((i % 1009 + 1) as f64))
        .sum();
    assert_eq!(got_sum, *sum, "{what} {w}x{h}: sum");
    assert_eq!(got_weighted, *weighted_sum, "{what} {w}x{h}: weighted sum");
    for &(x, y, v) in samples {
        assert_eq!(
            data[y * out.width() + x],
            v,
            "{what} {w}x{h}: pixel ({x}, {y})"
        );
    }
}

#[test]
fn pyrdown_f32_c1_matches_opencv() -> Result<(), ImageError> {
    for pins in &PYRDOWN {
        let (width, height) = pins.0;
        let src = synthetic_plane(ImageSize { width, height })?;
        let half = ImageSize {
            width: width.div_ceil(2),
            height: height.div_ceil(2),
        };
        let mut dst = Image::<f32, 1>::from_size_val(half, 0.0)?;
        pyrdown_f32(&src, &mut dst)?;
        assert_pins(&dst, pins, "pyrDown");
    }
    Ok(())
}

#[test]
fn scharr_float_c1_matches_opencv() -> Result<(), ImageError> {
    for (pins_dx, pins_dy) in SCHARR_DX.iter().zip(&SCHARR_DY) {
        let (width, height) = pins_dx.0;
        let size = ImageSize { width, height };
        let src = synthetic_plane(size)?;
        let mut dx = Image::<f32, 1>::from_size_val(size, 0.0)?;
        let mut dy = Image::<f32, 1>::from_size_val(size, 0.0)?;
        scharr_spatial_gradient_float(&src, &mut dx, &mut dy)?;
        assert_pins(&dx, pins_dx, "Scharr dx");
        assert_pins(&dy, pins_dy, "Scharr dy");
    }
    Ok(())
}
