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

/// Every `f32` encoding of a missing measurement.
const F32_INVALID: [f32; 6] = [0.0, -0.0, -1.0, f32::INFINITY, f32::NEG_INFINITY, f32::NAN];

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
fn flying_mask_ignores_every_f32_invalid_encoding() -> TestResult {
    // A 1.0 surface with a hole ring around a 1.5 outlier at the centre:
    // the outlier is flagged, the holes are never flagged nor counted.
    for bad in F32_INVALID {
        #[rustfmt::skip]
        let data = vec![
            1.0, 1.0, 1.0, 1.0, 1.0,
            1.0, bad, 1.0, bad, 1.0,
            1.0, 1.0, 1.5, 1.0, 1.0,
            1.0, bad, bad, 1.0, 1.0,
            1.0, 1.0, 1.0, 1.0, 1.0,
        ];
        let src = Image::<f32, 1>::new(sz(5, 5), data)?;
        let mut mask = Image::<u8, 1>::from_size_val(src.size(), 0)?;
        flying_pixel_mask(&src, &mut mask, 0.1, 0.0, 0)?;
        assert_eq!(flagged(&mask), vec![12], "{bad}");
        // 5 valid taps at the centre (4 holes): the holes are not counted.
        flying_pixel_mask(&src, &mut mask, 1.0, 0.0, 6)?;
        assert_eq!(flagged(&mask), vec![12], "{bad}");
        // Every pixel invalid: nothing to flag, even with min_valid > 9.
        let src = Image::<f32, 1>::from_size_val(sz(5, 5), bad)?;
        flying_pixel_mask(&src, &mut mask, 0.0, 0.0, 10)?;
        assert!(flagged(&mask).is_empty(), "{bad}");
    }
    Ok(())
}

#[test]
fn empty_images_are_a_no_op() -> TestResult {
    // Regression: zero-width images panicked in `par_chunks_mut(0)`.
    for size in [sz(0, 5), sz(5, 0), sz(0, 0)] {
        let mut mask = Image::<u8, 1>::from_size_val(size, 0)?;
        let src = Image::<u16, 1>::from_size_val(size, 1)?;
        flying_pixel_mask(&src, &mut mask, 1.0, 0.0, 0)?;
        let mut dst = Image::<u16, 1>::from_size_val(size, 0)?;
        invalidate_masked_depth(&src, &mask, &mut dst)?;
        let src = Image::<f32, 1>::from_size_val(size, 1.0)?;
        flying_pixel_mask(&src, &mut mask, 1.0, 0.0, 0)?;
        let mut dst = Image::<f32, 1>::from_size_val(size, 0.0)?;
        invalidate_masked_depth(&src, &mask, &mut dst)?;
        // Argument checks still run on an empty image.
        assert!(matches!(
            flying_pixel_mask(&src, &mut mask, f32::NAN, 0.0, 0),
            Err(ImageError::InvalidThreshold { .. })
        ));
    }
    Ok(())
}

#[test]
fn flying_mask_rejects_nan_and_negative_thresholds() -> TestResult {
    let src = Image::<u16, 1>::from_size_val(sz(4, 3), 1000)?;
    let src_f32 = Image::<f32, 1>::from_size_val(sz(4, 3), 1.0)?;
    let mut mask = Image::<u8, 1>::from_size_val(sz(4, 3), 7)?;
    for bad in [f32::NAN, -1.0, -0.5e-9, f32::NEG_INFINITY] {
        for (abs_t, rel_t, name) in [(bad, 0.0, "abs_threshold"), (0.0, bad, "rel_threshold")] {
            for result in [
                flying_pixel_mask(&src, &mut mask, abs_t, rel_t, 0),
                flying_pixel_mask(&src_f32, &mut mask, abs_t, rel_t, 0),
            ] {
                match result {
                    Err(ImageError::InvalidThreshold { name: got, value }) => {
                        assert_eq!(got, name);
                        assert_eq!(value.to_bits(), bad.to_bits());
                    }
                    other => panic!("{name} = {bad}: {other:?}"),
                }
            }
            // Rejected before the mask is written.
            assert!(mask.as_slice().iter().all(|&v| v == 7));
        }
    }
    // Zero, -0.0 (== 0) and +inf are usable thresholds.
    for ok in [0.0, -0.0, 1.0, f32::INFINITY] {
        flying_pixel_mask(&src, &mut mask, ok, ok, 0)?;
    }
    // +inf disables the deviation test: only `min_valid` flags.
    let mut data = vec![1000u16; 12];
    data[5] = 9000;
    let src = Image::<u16, 1>::new(sz(4, 3), data)?;
    flying_pixel_mask(&src, &mut mask, f32::INFINITY, 0.0, 0)?;
    assert!(flagged(&mask).is_empty());
    flying_pixel_mask(&src, &mut mask, 0.0, 0.0, 0)?;
    assert_eq!(flagged(&mask), vec![5]);
    Ok(())
}

#[test]
fn depth_ops_reject_size_mismatch() -> TestResult {
    // Same pixel count, different shape.
    let src = Image::<u16, 1>::from_size_val(sz(8, 4), 1)?;
    let mut dst = Image::<u16, 1>::from_size_val(sz(8, 4), 0)?;
    let mut mask = Image::<u8, 1>::from_size_val(sz(4, 8), 0)?;
    assert!(matches!(
        flying_pixel_mask(&src, &mut mask, 1.0, 0.0, 0),
        Err(ImageError::InvalidImageSize(..))
    ));
    let mask = Image::<u8, 1>::from_size_val(sz(4, 8), 0)?;
    assert!(matches!(
        invalidate_masked_depth(&src, &mask, &mut dst),
        Err(ImageError::InvalidImageSize(..))
    ));
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
