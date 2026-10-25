use super::*;
use kernels::DepthRepr;
use kornia_image::ImageSize;

type TestResult = Result<(), Box<dyn std::error::Error>>;

fn sz(width: usize, height: usize) -> ImageSize {
    ImageSize { width, height }
}

/// Indices of the non-zero mask entries.
fn flagged(mask: &Image<u8, 1>) -> Vec<usize> {
    mask.as_slice()
        .iter()
        .enumerate()
        .filter(|(_, &v)| v != 0)
        .map(|(i, _)| i)
        .collect()
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

fn random_f32(len: usize, seed: u64) -> Vec<f32> {
    let mut rng = Lcg(seed);
    (0..len)
        .map(|_| {
            let r = rng.next();
            if r.is_multiple_of(5) {
                f32::NAN
            } else {
                0.1 + (r >> 8) as f32 / 65536.0
            }
        })
        .collect()
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
    let mut mask = Image::<u8, 1>::from_size_val(sz(4, 8), 0)?;
    assert!(flying_pixel_mask(&src, &mut mask, 1.0, 0.0, 0).is_err());
    let mask = Image::<u8, 1>::from_size_val(sz(4, 8), 0)?;
    assert!(invalidate_masked_depth(&src, &mask, &mut dst).is_err());
    Ok(())
}

// ---- flying-pixel mask ------------------------------------------------------

const STEP_W: usize = 8;
const STEP_H: usize = 5;
const NEAR: u16 = 1000;
const FAR: u16 = 2000;
const ABS: f32 = 100.0;
const REL: f32 = 0.02;

/// Columns 0..4 at `NEAR`, 4..8 at `FAR`: a clean vertical step.
fn step() -> Vec<u16> {
    (0..STEP_W * STEP_H)
        .map(|i| if i % STEP_W < 4 { NEAR } else { FAR })
        .collect()
}

#[test]
fn flying_mask_clean_step_flags_nothing() -> TestResult {
    // Next to the step the window is 6 own-surface + 3 other-surface taps,
    // so the median is the pixel's own depth.
    let src = Image::<u16, 1>::new(sz(STEP_W, STEP_H), step())?;
    let mut mask = Image::<u8, 1>::from_size_val(src.size(), 7)?;
    flying_pixel_mask(&src, &mut mask, ABS, REL, 0)?;
    assert!(flagged(&mask).is_empty());
    Ok(())
}

#[test]
fn flying_mask_flags_lone_mixed_pixel_on_step() -> TestResult {
    // One in-between sample on the near side of the step: window = 5 near,
    // 1 mixed, 3 far -> median NEAR, |1500 - 1000| > max(100, 30).
    // A whole column of mixed samples would be a two-step staircase and
    // flag nothing, which is why the fixture uses a single pixel.
    let mut data = step();
    let at = 2 * STEP_W + 3;
    data[at] = 1500;
    let src = Image::<u16, 1>::new(sz(STEP_W, STEP_H), data)?;
    let mut mask = Image::<u8, 1>::from_size_val(src.size(), 0)?;
    flying_pixel_mask(&src, &mut mask, ABS, REL, 0)?;
    assert_eq!(flagged(&mask), vec![at]);
    assert_eq!(mask.as_slice()[at], 255);
    Ok(())
}

#[test]
fn flying_mask_flags_one_pixel_tooth() -> TestResult {
    // A far sample protruding one pixel into the near surface: window =
    // 5 near, 4 far -> median NEAR, |2000 - 1000| > 100. Its neighbours
    // keep a near majority (or a far one on the far side) and stay.
    let mut data = step();
    let at = 2 * STEP_W + 3;
    data[at] = FAR;
    let src = Image::<u16, 1>::new(sz(STEP_W, STEP_H), data)?;
    let mut mask = Image::<u8, 1>::from_size_val(src.size(), 0)?;
    flying_pixel_mask(&src, &mut mask, ABS, REL, 0)?;
    assert_eq!(flagged(&mask), vec![at]);
    Ok(())
}

#[test]
fn flying_mask_smooth_ramp_flags_nothing() -> TestResult {
    // Planar ramp, 20 units/column and 10 units/row: interior medians equal
    // the centre; replicated border windows deviate by at most 10.
    let (w, h) = (16, 12);
    let data: Vec<u16> = (0..w * h)
        .map(|i| 1000 + 20 * (i % w) as u16 + 10 * (i / w) as u16)
        .collect();
    let src = Image::<u16, 1>::new(sz(w, h), data.clone())?;
    let mut mask = Image::<u8, 1>::from_size_val(src.size(), 0)?;
    flying_pixel_mask(&src, &mut mask, 10.0, 0.0, 9)?;
    assert!(flagged(&mask).is_empty());
    // The border bound is tight: below 10 the replicated corner flags.
    flying_pixel_mask(&src, &mut mask, 9.0, 0.0, 0)?;
    assert!(flagged(&mask).contains(&0));

    let data: Vec<f32> = data.iter().map(|&v| f32::from(v) / 1000.0).collect();
    let src = Image::<f32, 1>::new(sz(w, h), data)?;
    flying_pixel_mask(&src, &mut mask, 0.0, 0.02, 0)?;
    assert!(flagged(&mask).is_empty());
    Ok(())
}

#[test]
fn flying_mask_relative_threshold() -> TestResult {
    // 3 % deviation on a 2000-unit surface: 60 units.
    let mut data = vec![2000u16; 25];
    data[12] = 2060;
    let src = Image::<u16, 1>::new(sz(5, 5), data)?;
    let mut mask = Image::<u8, 1>::from_size_val(src.size(), 0)?;
    flying_pixel_mask(&src, &mut mask, 0.0, 0.02, 0)?;
    assert_eq!(flagged(&mask), vec![12]);
    flying_pixel_mask(&src, &mut mask, 0.0, 0.05, 0)?;
    assert!(flagged(&mask).is_empty());
    // The absolute floor wins when it is larger than rel * d.
    flying_pixel_mask(&src, &mut mask, 100.0, 0.02, 0)?;
    assert!(flagged(&mask).is_empty());
    Ok(())
}

#[test]
fn flying_mask_borders_count_replicated_taps() -> TestResult {
    // Only the corner (0, 0) and the top-edge pixel (3, 0) are valid.
    // Corner window rows {0,0,1} x cols {0,0,1}: the corner tap appears 4
    // times. Edge window rows {0,0,1} x cols {2,3,4}: the pixel appears twice.
    let (w, h) = (6, 4);
    let mut data = vec![0u16; w * h];
    data[0] = 1000;
    data[3] = 1000;
    let src = Image::<u16, 1>::new(sz(w, h), data)?;
    let mut mask = Image::<u8, 1>::from_size_val(src.size(), 0)?;
    flying_pixel_mask(&src, &mut mask, ABS, REL, 2)?;
    assert!(flagged(&mask).is_empty());
    flying_pixel_mask(&src, &mut mask, ABS, REL, 3)?;
    assert_eq!(flagged(&mask), vec![3]);
    flying_pixel_mask(&src, &mut mask, ABS, REL, 4)?;
    assert_eq!(flagged(&mask), vec![3]);
    flying_pixel_mask(&src, &mut mask, ABS, REL, 5)?;
    assert_eq!(flagged(&mask), vec![0, 3]);
    // Above the tap count every valid pixel is flagged; invalid ones never.
    flying_pixel_mask(&src, &mut mask, ABS, REL, 10)?;
    assert_eq!(flagged(&mask), vec![0, 3]);
    Ok(())
}

#[test]
fn flying_mask_all_invalid_flags_nothing() -> TestResult {
    let src = Image::<u16, 1>::from_size_val(sz(9, 6), 0)?;
    let mut mask = Image::<u8, 1>::from_size_val(src.size(), 255)?;
    flying_pixel_mask(&src, &mut mask, 0.0, 0.0, 9)?;
    assert!(flagged(&mask).is_empty());

    let src = Image::<f32, 1>::from_size_val(sz(9, 6), f32::NAN)?;
    flying_pixel_mask(&src, &mut mask, 0.0, 0.0, 9)?;
    assert!(flagged(&mask).is_empty());
    Ok(())
}

#[test]
fn flying_mask_nan_neighbours_are_excluded() -> TestResult {
    // Valid 1.0 m surface with NaN holes around the centre; the centre is a
    // 1.5 m outlier. Its valid window samples: four at 1.0, itself -> flag.
    let n = f32::NAN;
    #[rustfmt::skip]
    let data = vec![
        1.0, 1.0, 1.0, 1.0, 1.0,
        1.0, n,   1.0, n,   1.0,
        1.0, 1.0, 1.5, 1.0, 1.0,
        1.0, n,   n,   1.0, 1.0,
        1.0, 1.0, 1.0, 1.0, 1.0,
    ];
    let src = Image::<f32, 1>::new(sz(5, 5), data)?;
    let mut mask = Image::<u8, 1>::from_size_val(src.size(), 0)?;
    flying_pixel_mask(&src, &mut mask, 0.1, 0.0, 0)?;
    assert_eq!(flagged(&mask), vec![12]);
    // Support test: the centre has 5 valid taps; every other valid pixel
    // has at least 6.
    flying_pixel_mask(&src, &mut mask, 1.0, 0.0, 6)?;
    assert_eq!(flagged(&mask), vec![12]);
    flying_pixel_mask(&src, &mut mask, 1.0, 0.0, 5)?;
    assert!(flagged(&mask).is_empty());
    Ok(())
}

#[test]
fn invalidate_masked_depth_writes_invalid_marker() -> TestResult {
    let mut data = step();
    data[2 * STEP_W + 3] = 1500;
    let src = Image::<u16, 1>::new(sz(STEP_W, STEP_H), data)?;
    let mut mask = Image::<u8, 1>::from_size_val(src.size(), 0)?;
    flying_pixel_mask(&src, &mut mask, ABS, REL, 0)?;
    let mut dst = Image::<u16, 1>::from_size_val(src.size(), 1)?;
    invalidate_masked_depth(&src, &mask, &mut dst)?;
    let mut expect = step();
    expect[2 * STEP_W + 3] = 0;
    assert_eq!(dst.as_slice(), &expect[..]);

    let src = Image::<f32, 1>::new(sz(3, 1), vec![1.0, 2.0, 3.0])?;
    let mask = Image::<u8, 1>::new(sz(3, 1), vec![1, 0, 255])?;
    let mut dst = Image::<f32, 1>::from_size_val(src.size(), 0.0)?;
    invalidate_masked_depth(&src, &mask, &mut dst)?;
    let d = dst.as_slice();
    assert!(d[0].is_nan() && d[1] == 2.0 && d[2].is_nan());
    Ok(())
}
