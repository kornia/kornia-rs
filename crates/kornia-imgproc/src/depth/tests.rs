use super::*;
use kernels::DepthRepr;
use kornia_image::ImageSize;

type TestResult = Result<(), Box<dyn std::error::Error>>;

fn sz(width: usize, height: usize) -> ImageSize {
    ImageSize { width, height }
}

/// Straightforward oracle: gather the replicate-border window, keep the
/// valid samples, sort, take rank `n / 2`.
fn naive_median<T: DepthRepr + PartialOrd, const C: usize>(
    src: &[T],
    w: usize,
    h: usize,
    ksize: usize,
) -> Vec<T> {
    let r = (ksize / 2) as isize;
    let mut out = vec![T::INVALID; src.len()];
    for y in 0..h {
        for x in 0..w {
            for c in 0..C {
                let mut valid = Vec::new();
                for dy in -r..=r {
                    for dx in -r..=r {
                        let sy = (y as isize + dy).clamp(0, h as isize - 1) as usize;
                        let sx = (x as isize + dx).clamp(0, w as isize - 1) as usize;
                        let v = src[(sy * w + sx) * C + c];
                        if v.is_valid() {
                            valid.push(v);
                        }
                    }
                }
                valid.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
                if !valid.is_empty() {
                    out[(y * w + x) * C + c] = valid[valid.len() / 2];
                }
            }
        }
    }
    out
}

/// Deterministic LCG.
struct Lcg(u64);
impl Lcg {
    fn next(&mut self) -> u32 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (self.0 >> 33) as u32
    }
}

fn random_u16(len: usize, seed: u64) -> Vec<u16> {
    let mut rng = Lcg(seed);
    (0..len)
        .map(|_| {
            let r = rng.next();
            if r.is_multiple_of(5) {
                0
            } else {
                (r >> 8) as u16
            }
        })
        .collect()
}

/// Every `f32` encoding of a missing measurement.
const F32_INVALID: [f32; 6] = [0.0, -0.0, -1.0, f32::INFINITY, f32::NEG_INFINITY, f32::NAN];

/// ~20 % holes, cycling through every invalid encoding.
fn random_f32(len: usize, seed: u64) -> Vec<f32> {
    let mut rng = Lcg(seed);
    (0..len)
        .map(|_| {
            let r = rng.next();
            if r.is_multiple_of(5) {
                F32_INVALID[(r >> 8) as usize % F32_INVALID.len()]
            } else {
                0.1 + (r >> 8) as f32 / 65536.0
            }
        })
        .collect()
}

#[test]
fn validity_rule() {
    assert!(!is_valid_depth_u16(0));
    for d in [1u16, 1000, u16::MAX] {
        assert!(is_valid_depth_u16(d));
    }
    for d in F32_INVALID {
        assert!(!is_valid_depth_f32(d), "{d}");
    }
    for d in [f32::MIN_POSITIVE, 1e-6, 1.0, 1e6, f32::MAX] {
        assert!(is_valid_depth_f32(d), "{d}");
    }
    // The kernels' rule is the public one.
    for d in [0u16, 1, u16::MAX] {
        assert_eq!(d.is_valid(), is_valid_depth_u16(d));
    }
    for d in F32_INVALID.into_iter().chain([1.0, f32::MAX]) {
        assert_eq!(d.is_valid(), is_valid_depth_f32(d));
    }
}

#[test]
fn median_blur_depth_excludes_every_f32_invalid_encoding() -> TestResult {
    // Same layout as `median_blur_depth_excludes_nan`, once per encoding.
    for bad in F32_INVALID {
        let src =
            Image::<f32, 1>::new(sz(3, 3), vec![1.0, bad, 3.0, bad, bad, bad, 2.0, bad, 4.0])?;
        for ksize in [3, 5] {
            let mut dst = Image::<f32, 1>::from_size_val(src.size(), 0.0)?;
            median_blur_depth(&src, &mut dst, ksize)?;
            // Centre: valid {1, 2, 3, 4} -> rank 2 = 3.
            assert_eq!(dst.as_slice()[4], 3.0, "{bad} k={ksize}");
            assert!(
                dst.as_slice().iter().all(|&v| is_valid_depth_f32(v)),
                "{bad} k={ksize}: {:?}",
                dst.as_slice()
            );
        }
        // An all-invalid window writes the canonical NaN marker.
        let src = Image::<f32, 1>::from_size_val(sz(4, 3), bad)?;
        let mut dst = Image::<f32, 1>::from_size_val(src.size(), 1.0)?;
        median_blur_depth(&src, &mut dst, 3)?;
        assert!(dst.as_slice().iter().all(|v| v.is_nan()), "{bad}");
    }
    Ok(())
}

// Fixtures from OpenCV 4.8.0 (`cv2.medianBlur`, default BORDER_REPLICATE):
//   rng = np.random.default_rng(7); h, w = 6, 7
//   u = rng.integers(1, 65536, size=(h, w), dtype=np.uint16)
//   f = (rng.random((h, w), dtype=np.float32) * 10.0).round(3).astype(np.float32)
//   cv2.medianBlur(u, k), cv2.medianBlur(f, k) for k in (3, 5)
// No sample is invalid, so the valid-aware median must equal OpenCV.
const CV_W: usize = 7;
const CV_H: usize = 6;
#[rustfmt::skip]
const CV_U16_SRC: [u16; 42] = [19083, 61925, 16809, 40966, 27285, 44838, 52668, 58799, 63842, 37898, 22111, 50835, 10050, 54634, 11691, 14759, 20805, 3639, 45721, 19671, 10036, 18682, 13015, 57249, 44144, 59809, 4390, 345, 1757, 32754, 1679, 53820, 6494, 8614, 48633, 52236, 14625, 7804, 38343, 30666, 56392, 53507];
#[rustfmt::skip]
const CV_U16_K3: [u16; 42] = [58799, 37898, 40966, 27285, 40966, 44838, 52668, 19083, 20805, 22111, 27285, 27285, 44838, 44838, 18682, 20805, 22111, 44144, 22111, 19671, 10036, 13015, 14759, 20805, 44144, 19671, 10036, 10036, 18682, 14625, 32754, 38343, 38343, 30666, 48633, 32754, 14625, 14625, 30666, 38343, 48633, 53507];
#[rustfmt::skip]
const CV_U16_K5: [u16; 42] = [19083, 19083, 27285, 37898, 37898, 44838, 50835, 19083, 19083, 27285, 37898, 37898, 40966, 44838, 18682, 19083, 27285, 27285, 27285, 40966, 44838, 18682, 20805, 30666, 22111, 30666, 38343, 45721, 14759, 18682, 20805, 20805, 30666, 38343, 45721, 18682, 32754, 30666, 30666, 38343, 48633, 53507];
#[rustfmt::skip]
const CV_F32_SRC: [f32; 42] = [3.03, 3.416, 2.784, 7.195, 2.549, 9.905, 4.451, 4.781, 5.045, 5.825, 5.535, 5.094, 9.955, 8.077, 7.927, 7.003, 6.222, 3.41, 9.89, 4.662, 2.153, 8.451, 1.602, 8.574, 6.125, 1.147, 0.439, 4.448, 0.357, 1.416, 5.149, 9.703, 4.662, 8.084, 9.172, 8.235, 6.292, 4.414, 5.141, 2.663, 4.969, 3.794];
#[rustfmt::skip]
const CV_F32_K3: [f32; 42] = [3.416, 3.416, 5.045, 5.094, 7.195, 5.094, 8.077, 4.781, 5.045, 5.535, 5.535, 5.535, 5.094, 4.662, 7.003, 6.222, 5.825, 5.825, 5.094, 4.662, 4.448, 7.003, 6.222, 6.125, 6.125, 4.662, 4.662, 4.448, 6.292, 5.149, 5.149, 5.141, 4.969, 4.448, 4.448, 6.292, 5.149, 5.141, 4.662, 4.969, 4.662, 4.969];
#[rustfmt::skip]
const CV_F32_K5: [f32; 42] = [3.416, 3.416, 3.416, 5.094, 5.094, 4.662, 4.451, 4.781, 5.045, 5.045, 5.094, 5.094, 4.451, 4.451, 4.781, 5.149, 5.094, 5.149, 5.535, 5.094, 4.662, 6.222, 5.825, 5.149, 5.141, 5.141, 4.969, 4.662, 7.927, 6.292, 5.149, 4.969, 4.662, 4.448, 3.794, 8.235, 6.292, 5.141, 4.969, 4.662, 4.448, 3.794];

#[test]
fn median_blur_depth_matches_opencv_u16() -> TestResult {
    let src = Image::<u16, 1>::new(sz(CV_W, CV_H), CV_U16_SRC.to_vec())?;
    let mut dst = Image::<u16, 1>::from_size_val(src.size(), 0)?;
    median_blur_depth(&src, &mut dst, 3)?;
    assert_eq!(dst.as_slice(), &CV_U16_K3);
    median_blur_depth(&src, &mut dst, 5)?;
    assert_eq!(dst.as_slice(), &CV_U16_K5);
    Ok(())
}

#[test]
fn median_blur_depth_matches_opencv_f32() -> TestResult {
    let src = Image::<f32, 1>::new(sz(CV_W, CV_H), CV_F32_SRC.to_vec())?;
    let mut dst = Image::<f32, 1>::from_size_val(src.size(), 0.0)?;
    median_blur_depth(&src, &mut dst, 3)?;
    assert_eq!(dst.as_slice(), &CV_F32_K3);
    median_blur_depth(&src, &mut dst, 5)?;
    assert_eq!(dst.as_slice(), &CV_F32_K5);
    Ok(())
}

#[test]
fn median_blur_depth_excludes_zero() -> TestResult {
    // 3×3, replicate border. Centre window = all 9 pixels:
    // valid {10, 11, 12, 13, 14, 15}, n = 6 -> upper median rank 3 = 13.
    // Top-left window (rows 0,0,1 x cols 0,0,1): 10,10,0,10,10,0,11,11,0
    // -> valid {10,10,10,10,11,11}, rank 3 = 10.
    let src = Image::<u16, 1>::new(sz(3, 3), vec![10, 0, 12, 11, 0, 13, 0, 14, 15])?;
    let mut dst = Image::<u16, 1>::from_size_val(src.size(), 0)?;
    median_blur_depth(&src, &mut dst, 3)?;
    assert_eq!(dst.as_slice()[4], 13);
    assert_eq!(dst.as_slice()[0], 10);
    assert!(dst.as_slice().iter().all(|&v| v != 0));
    Ok(())
}

#[test]
fn median_blur_depth_excludes_nan() -> TestResult {
    // NaN centre with valid neighbours is filled; NaN neighbours never
    // reach the median.
    let n = f32::NAN;
    let src = Image::<f32, 1>::new(sz(3, 3), vec![1.0, n, 3.0, n, n, n, 2.0, n, 4.0])?;
    let mut dst = Image::<f32, 1>::from_size_val(src.size(), 0.0)?;
    median_blur_depth(&src, &mut dst, 3)?;
    // Centre: valid {1, 2, 3, 4} -> rank 2 = 3.
    assert_eq!(dst.as_slice()[4], 3.0);
    // Top-middle (rows 0,0,1 x cols 0,1,2): 1,n,3,1,n,3,n,n,n -> {1,1,3,3} -> 3.
    assert_eq!(dst.as_slice()[1], 3.0);
    assert!(dst.as_slice().iter().all(|v| !v.is_nan()));
    Ok(())
}

#[test]
fn median_blur_depth_all_invalid_stays_invalid() -> TestResult {
    for ksize in [3, 5] {
        let src = Image::<u16, 1>::from_size_val(sz(11, 7), 0)?;
        let mut dst = Image::<u16, 1>::from_size_val(src.size(), 7)?;
        median_blur_depth(&src, &mut dst, ksize)?;
        assert!(dst.as_slice().iter().all(|&v| v == 0));

        let src = Image::<f32, 3>::from_size_val(sz(11, 7), f32::NAN)?;
        let mut dst = Image::<f32, 3>::from_size_val(src.size(), 1.0)?;
        median_blur_depth(&src, &mut dst, ksize)?;
        assert!(dst.as_slice().iter().all(|v| v.is_nan()));
    }
    Ok(())
}

#[test]
fn median_blur_depth_isolated_invalid_region_stays_invalid() -> TestResult {
    // A 7×7 hole: with k=3 the 5×5 core has no valid tap; its rim does.
    let (w, h) = (9, 9);
    let mut data = vec![500u16; w * h];
    for y in 1..8 {
        for x in 1..8 {
            data[y * w + x] = 0;
        }
    }
    let src = Image::<u16, 1>::new(sz(w, h), data)?;
    let mut dst = Image::<u16, 1>::from_size_val(src.size(), 0)?;
    median_blur_depth(&src, &mut dst, 3)?;
    for y in 0..h {
        for x in 0..w {
            let core = (2..7).contains(&x) && (2..7).contains(&y);
            let expect = if core { 0 } else { 500 };
            assert_eq!(dst.as_slice()[y * w + x], expect, "({x}, {y})");
        }
    }
    Ok(())
}

#[test]
fn median_blur_depth_matches_naive_on_random() -> TestResult {
    for (w, h) in [(1, 1), (2, 9), (5, 4), (17, 13), (40, 3)] {
        for ksize in [3, 5] {
            let data = random_u16(w * h, 1 + w as u64);
            let expect = naive_median::<u16, 1>(&data, w, h, ksize);
            let src = Image::<u16, 1>::new(sz(w, h), data)?;
            let mut dst = Image::<u16, 1>::from_size_val(src.size(), 0)?;
            median_blur_depth(&src, &mut dst, ksize)?;
            assert_eq!(dst.as_slice(), &expect[..], "u16 C1 {w}x{h} k={ksize}");

            let data = random_u16(w * h * 3, 2 + h as u64);
            let expect = naive_median::<u16, 3>(&data, w, h, ksize);
            let src = Image::<u16, 3>::new(sz(w, h), data)?;
            let mut dst = Image::<u16, 3>::from_size_val(src.size(), 0)?;
            median_blur_depth(&src, &mut dst, ksize)?;
            assert_eq!(dst.as_slice(), &expect[..], "u16 C3 {w}x{h} k={ksize}");

            let data = random_f32(w * h, 3 + w as u64);
            let expect = naive_median::<f32, 1>(&data, w, h, ksize);
            let src = Image::<f32, 1>::new(sz(w, h), data)?;
            let mut dst = Image::<f32, 1>::from_size_val(src.size(), 0.0)?;
            median_blur_depth(&src, &mut dst, ksize)?;
            let got: Vec<u32> = dst.as_slice().iter().map(|v| v.to_bits()).collect();
            let want: Vec<u32> = expect.iter().map(|v| v.to_bits()).collect();
            assert_eq!(got, want, "f32 C1 {w}x{h} k={ksize}");
        }
    }
    Ok(())
}

#[test]
fn empty_images_are_a_no_op() -> TestResult {
    // Regression: zero-width images panicked in `par_chunks_mut(0)`.
    for size in [sz(0, 5), sz(5, 0), sz(0, 0)] {
        for ksize in [3, 5] {
            let src = Image::<u16, 1>::from_size_val(size, 1)?;
            let mut dst = Image::<u16, 1>::from_size_val(size, 0)?;
            median_blur_depth(&src, &mut dst, ksize)?;
            let src = Image::<u16, 3>::from_size_val(size, 1)?;
            let mut dst = Image::<u16, 3>::from_size_val(size, 0)?;
            median_blur_depth(&src, &mut dst, ksize)?;
            let src = Image::<f32, 1>::from_size_val(size, 1.0)?;
            let mut dst = Image::<f32, 1>::from_size_val(size, 0.0)?;
            median_blur_depth(&src, &mut dst, ksize)?;
        }
        // Argument checks still run on an empty image.
        let src = Image::<u16, 1>::from_size_val(size, 1)?;
        let mut dst = Image::<u16, 1>::from_size_val(size, 0)?;
        assert!(matches!(
            median_blur_depth(&src, &mut dst, 4),
            Err(ImageError::InvalidKernelLength(..))
        ));
    }
    Ok(())
}

#[test]
fn median_blur_depth_rejects_bad_args() -> TestResult {
    let src = Image::<u16, 1>::from_size_val(sz(8, 4), 1)?;
    let mut dst = Image::<u16, 1>::from_size_val(sz(8, 4), 0)?;
    for ksize in [1, 2, 4, 7] {
        assert!(matches!(
            median_blur_depth(&src, &mut dst, ksize),
            Err(ImageError::InvalidKernelLength(..))
        ));
    }
    // Same pixel count, different shape.
    let mut transposed = Image::<u16, 1>::from_size_val(sz(4, 8), 0)?;
    assert!(matches!(
        median_blur_depth(&src, &mut transposed, 3),
        Err(ImageError::InvalidImageSize(..))
    ));
    Ok(())
}
