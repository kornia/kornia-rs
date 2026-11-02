use super::*;
use kornia_image::ImageSize;

type TestResult = Result<(), Box<dyn std::error::Error>>;

const CONNECTIVITIES: [Connectivity; 2] = [Connectivity::Four, Connectivity::Eight];

fn sz(width: usize, height: usize) -> ImageSize {
    ImageSize { width, height }
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

/// Blocky regions (so that not every pixel is near a boundary) with
/// sparse single-pixel speckle of a random label.
fn labels_u16(w: usize, h: usize, seed: u64) -> Vec<u16> {
    let mut rng = Lcg(seed);
    let mut out = Vec::with_capacity(w * h);
    for y in 0..h {
        for x in 0..w {
            let r = rng.next();
            let region = ((x / 6) ^ (y / 5)) % 4;
            out.push(if r.is_multiple_of(23) {
                (r >> 8) as u16 % 7
            } else {
                1000 + region as u16
            });
        }
    }
    out
}

fn labels_u8(w: usize, h: usize, seed: u64) -> Vec<u8> {
    labels_u16(w, h, seed)
        .into_iter()
        .map(|v| (v % 251) as u8)
        .collect()
}

/// The literal definition: `p` is filled when a pixel with another label
/// lies within `radius` under the connectivity's metric. O(radius²) per
/// pixel; the radius is clamped to `w + h` (no farther pixel exists).
fn oracle<T: Copy + PartialEq>(
    src: &[T],
    w: usize,
    h: usize,
    radius: usize,
    connectivity: Connectivity,
    fill: T,
) -> Vec<T> {
    let r = radius.min(w + h) as isize;
    let mut out = Vec::with_capacity(src.len());
    for y in 0..h as isize {
        for x in 0..w as isize {
            let v = src[y as usize * w + x as usize];
            let mut hit = false;
            // Offsets clipped to the image: farther pixels do not exist.
            for dy in (-r).max(-y)..=r.min(h as isize - 1 - y) {
                for dx in (-r).max(-x)..=r.min(w as isize - 1 - x) {
                    let d = match connectivity {
                        Connectivity::Four => dx.abs() + dy.abs(),
                        Connectivity::Eight => dx.abs().max(dy.abs()),
                    };
                    let (sx, sy) = (x + dx, y + dy);
                    if d > r || sx < 0 || sy < 0 || sx >= w as isize || sy >= h as isize {
                        continue;
                    }
                    hit |= src[sy as usize * w + sx as usize] != v;
                }
            }
            out.push(if hit { fill } else { v });
        }
    }
    out
}

fn run<T: LabelPixel>(
    data: Vec<T>,
    w: usize,
    h: usize,
    radius: usize,
    connectivity: Connectivity,
    fill: T,
) -> Result<Vec<T>, ImageError> {
    let src = Image::<T, 1>::new(sz(w, h), data)?;
    let mut dst = Image::<T, 1>::from_size_val(src.size(), fill)?;
    erode_label_boundaries(&src, &mut dst, radius, connectivity, fill)?;
    Ok(dst.as_slice().to_vec())
}

fn check_against_oracle<T: LabelPixel + std::fmt::Debug>(
    data: Vec<T>,
    w: usize,
    h: usize,
    radius: usize,
    connectivity: Connectivity,
    fill: T,
) -> TestResult {
    let expected = oracle(&data, w, h, radius, connectivity, fill);
    let got = run(data, w, h, radius, connectivity, fill)?;
    assert_eq!(got, expected, "{w}x{h} r={radius} {connectivity:?}");
    Ok(())
}

/// Vertical two-label stripe: columns `< split` are `a`, the rest `b`.
fn stripe(w: usize, h: usize, split: usize, a: u8, b: u8) -> Vec<u8> {
    (0..w * h)
        .map(|i| if i % w < split { a } else { b })
        .collect()
}

#[test]
fn radius_zero_is_identity() -> TestResult {
    for connectivity in CONNECTIVITIES {
        let data = labels_u8(37, 23, 1);
        assert_eq!(run(data.clone(), 37, 23, 0, connectivity, 0)?, data);
        let data = labels_u16(37, 23, 2);
        assert_eq!(run(data.clone(), 37, 23, 0, connectivity, 0)?, data);
    }
    Ok(())
}

#[test]
fn single_label_is_unchanged() -> TestResult {
    for connectivity in CONNECTIVITIES {
        for radius in [1, 2, 7, 1000] {
            let data = vec![5u8; 13 * 9];
            assert_eq!(run(data.clone(), 13, 9, radius, connectivity, 0)?, data);
            let data = vec![700u16; 13 * 9];
            assert_eq!(run(data.clone(), 13, 9, radius, connectivity, 0)?, data);
        }
    }
    Ok(())
}

#[test]
fn stripe_radius_one_marks_both_sides() -> TestResult {
    // 6x3, labels 1 | 2 split after column 2; radius 1 marks columns 2 and
    // 3 on every row (both sides), never the image border columns.
    for connectivity in CONNECTIVITIES {
        let got = run(stripe(6, 3, 3, 1, 2), 6, 3, 1, connectivity, 9)?;
        #[rustfmt::skip]
        let expected: Vec<u8> = vec![
            1, 1, 9, 9, 2, 2,
            1, 1, 9, 9, 2, 2,
            1, 1, 9, 9, 2, 2,
        ];
        assert_eq!(got, expected, "{connectivity:?}");
    }
    Ok(())
}

#[test]
fn stripe_radius_two_and_n() -> TestResult {
    for connectivity in CONNECTIVITIES {
        let got = run(stripe(8, 2, 4, 1, 2), 8, 2, 2, connectivity, 0)?;
        #[rustfmt::skip]
        let expected: Vec<u8> = vec![
            1, 1, 0, 0, 0, 0, 2, 2,
            1, 1, 0, 0, 0, 0, 2, 2,
        ];
        assert_eq!(got, expected, "r=2 {connectivity:?}");

        // A radius reaching across the image marks every pixel, whatever
        // its size (no overflow, no error).
        for radius in [8, 1000, usize::MAX] {
            let got = run(stripe(8, 2, 4, 1, 2), 8, 2, radius, connectivity, 0)?;
            assert_eq!(got, vec![0u8; 16], "r={radius} {connectivity:?}");
        }
    }
    Ok(())
}

#[test]
fn single_pixel_island_shapes() -> TestResult {
    // 7x7 of label 1 with a label-2 pixel at the centre: radius r marks the
    // Manhattan diamond (Four) or the Chebyshev square (Eight) of radius r
    // around it.
    let (w, h, c) = (7usize, 7usize, 3isize);
    let mut data = vec![1u8; w * h];
    data[3 * w + 3] = 2;
    for radius in [1usize, 2, 3] {
        for connectivity in CONNECTIVITIES {
            let got = run(data.clone(), w, h, radius, connectivity, 0)?;
            for y in 0..h as isize {
                for x in 0..w as isize {
                    let (dx, dy) = ((x - c).abs(), (y - c).abs());
                    let d = match connectivity {
                        Connectivity::Four => dx + dy,
                        Connectivity::Eight => dx.max(dy),
                    };
                    let expected = if d <= radius as isize { 0 } else { 1 };
                    assert_eq!(
                        got[y as usize * w + x as usize],
                        expected,
                        "({x},{y}) r={radius} {connectivity:?}"
                    );
                }
            }
        }
    }
    Ok(())
}

#[test]
fn corner_contact_needs_eight_connectivity() -> TestResult {
    // Pixel (1, 1) touches the label-2 block only through its corner:
    // 4-connectivity leaves it, 8-connectivity marks it.
    #[rustfmt::skip]
    let data: Vec<u8> = vec![
        1, 1, 1, 1,
        1, 1, 1, 1,
        1, 1, 2, 2,
        1, 1, 2, 2,
    ];
    let four = run(data.clone(), 4, 4, 1, Connectivity::Four, 0)?;
    #[rustfmt::skip]
    let expected_four: Vec<u8> = vec![
        1, 1, 1, 1,
        1, 1, 0, 0,
        1, 0, 0, 0,
        1, 0, 0, 2,
    ];
    assert_eq!(four, expected_four);
    let eight = run(data, 4, 4, 1, Connectivity::Eight, 0)?;
    #[rustfmt::skip]
    let expected_eight: Vec<u8> = vec![
        1, 1, 1, 1,
        1, 0, 0, 0,
        1, 0, 0, 0,
        1, 0, 0, 2,
    ];
    assert_eq!(eight, expected_eight);
    Ok(())
}

#[test]
fn border_is_not_a_boundary() -> TestResult {
    // A region touching every image edge, with a single different pixel in
    // a corner: only the corner neighbourhood is marked, never the rest of
    // the border.
    for connectivity in CONNECTIVITIES {
        let mut data = vec![3u16; 5 * 4];
        data[0] = 4;
        let got = run(data, 5, 4, 1, connectivity, 0)?;
        let marked: Vec<usize> = (0..got.len()).filter(|&i| got[i] == 0).collect();
        let expected = match connectivity {
            Connectivity::Four => vec![0, 1, 5],
            Connectivity::Eight => vec![0, 1, 5, 6],
        };
        assert_eq!(marked, expected, "{connectivity:?}");
    }
    Ok(())
}

#[test]
fn matches_oracle_random_u8_u16() -> TestResult {
    // Sizes cover a single pixel, single row / column, widths below and
    // above the 8-lane NEON width with tails, and heights above one band.
    let sizes = [(1, 1), (1, 7), (7, 1), (2, 2), (3, 5), (33, 17), (70, 45)];
    for (i, &(w, h)) in sizes.iter().enumerate() {
        for radius in [0, 1, 2, 3, 5, 9] {
            for connectivity in CONNECTIVITIES {
                let seed = (i * 100 + radius) as u64;
                check_against_oracle(labels_u8(w, h, seed), w, h, radius, connectivity, 255)?;
                check_against_oracle(labels_u16(w, h, seed), w, h, radius, connectivity, 0)?;
            }
        }
    }
    Ok(())
}

#[test]
fn matches_oracle_radius_larger_than_band() -> TestResult {
    // A threshold above BAND_MIN_ROWS makes the bands `t` rows tall and the
    // band halo span several bands.
    let (w, h) = (23, 97);
    let mut data = vec![1u8; w * h];
    data[50 * w + 11] = 2;
    data[3 * w + 22] = 3;
    for radius in [kernels::BAND_MIN_ROWS + 5, 45] {
        for connectivity in CONNECTIVITIES {
            check_against_oracle(data.clone(), w, h, radius, connectivity, 0)?;
        }
    }
    Ok(())
}

#[test]
fn matches_oracle_parallel_path() -> TestResult {
    // Above PAR_MIN_PIXELS the bands run on rayon.
    let (w, h) = (300, 240);
    assert!(w * h >= kernels::PAR_MIN_PIXELS);
    for radius in [1, 3] {
        for connectivity in CONNECTIVITIES {
            check_against_oracle(labels_u16(w, h, 7), w, h, radius, connectivity, 0)?;
        }
    }
    Ok(())
}

/// `kernel` applied to a block of `block_rows` rows of width `len` equals
/// the scalar scan of each row, and the scalar scan equals the brute-force
/// definition.
fn check_horizontal_scan<D: kernels::Dist + std::fmt::Debug>(
    near_max: D,
    max: D,
    kernel: fn(&mut [D], usize),
    block_rows: usize,
) {
    let mut rng = Lcg(42);
    for len in 0..70 {
        for trial in 0..8 {
            // Mix of small distances, caps and the saturation edge.
            let alt = if trial % 2 == 0 {
                D::from_u16(7)
            } else {
                near_max
            };
            let row: Vec<D> = (0..len)
                .map(|_| match rng.next() % 4 {
                    0 => D::ZERO,
                    1 => D::from_u16((rng.next() % 9) as u16),
                    2 => max,
                    _ => alt,
                })
                .collect();
            let mut scalar = row.clone();
            kernels::horizontal_min_plus_scalar(&mut scalar);
            // Reference: min over x' of row[x'] + |x - x'|, saturating.
            let reference: Vec<D> = (0..len)
                .map(|x: usize| {
                    (0..len)
                        .map(|k: usize| {
                            let mut v = row[k];
                            for _ in 0..x.abs_diff(k).min(usize::from(u16::MAX)) {
                                if v == max {
                                    break;
                                }
                                v = v.sat_inc();
                            }
                            v
                        })
                        .min()
                        .unwrap_or(max)
                })
                .collect();
            assert_eq!(scalar, reference, "len={len} trial={trial}");
            // `block_rows` rows in one call: each row equals its own
            // scalar scan.
            let mut block: Vec<D> = (0..block_rows)
                .flat_map(|k| {
                    let mut r = row.clone();
                    r.rotate_left(k % len.max(1));
                    r
                })
                .collect();
            let expected: Vec<D> = block
                .chunks(len.max(1))
                .flat_map(|r| {
                    let mut r = r.to_vec();
                    kernels::horizontal_min_plus_scalar(&mut r);
                    r
                })
                .collect();
            kernel(&mut block, len);
            assert_eq!(
                block, expected,
                "rows={block_rows} len={len} trial={trial} row={row:?}"
            );
        }
    }
}

/// The dispatched scan (NEON on aarch64, scalar elsewhere) of one row, and
/// of an interleaved group plus a remainder, equals the scalar scan.
#[test]
fn horizontal_scan_dispatch_matches_scalar_u16() {
    let k: fn(&mut [u16], usize) = kernels::horizontal_min_plus_rows::<u16>;
    check_horizontal_scan::<u16>(u16::MAX - 1, u16::MAX, k, 1);
    check_horizontal_scan::<u16>(u16::MAX - 1, u16::MAX, k, kernels::H_ROWS + 1);
}

#[test]
fn horizontal_scan_dispatch_matches_scalar_u8() {
    let k: fn(&mut [u8], usize) = kernels::horizontal_min_plus_rows::<u8>;
    check_horizontal_scan::<u8>(u8::MAX - 1, u8::MAX, k, 1);
    check_horizontal_scan::<u8>(u8::MAX - 1, u8::MAX, k, kernels::H_ROWS + 1);
}

/// The NEON kernels, called directly (one row, and `H_ROWS` rows in lock
/// step), equal the scalar scan bit for bit.
#[cfg(target_arch = "aarch64")]
#[test]
fn horizontal_scan_neon_matches_scalar_u16() {
    use kernels::{horizontal_min_plus_neon_u16 as neon, H_ROWS};
    check_horizontal_scan::<u16>(u16::MAX - 1, u16::MAX, neon::<1>, 1);
    check_horizontal_scan::<u16>(u16::MAX - 1, u16::MAX, neon::<H_ROWS>, H_ROWS);
}

#[cfg(target_arch = "aarch64")]
#[test]
fn horizontal_scan_neon_matches_scalar_u8() {
    use kernels::{horizontal_min_plus_neon_u8 as neon, H_ROWS};
    check_horizontal_scan::<u8>(u8::MAX - 1, u8::MAX, neon::<1>, 1);
    check_horizontal_scan::<u8>(u8::MAX - 1, u8::MAX, neon::<H_ROWS>, H_ROWS);
}

/// The direct window minimum and the full two-sweep scan select the same
/// pixels (`<= t`) for every threshold the window handles, including rows
/// shorter than the window.
#[test]
fn horizontal_window_matches_scan_below_threshold() {
    let mut rng = Lcg(5);
    for t in 1..=kernels::DIRECT_MAX_THRESHOLD {
        let cap = (t + 1) as u8;
        for len in 1..70 {
            for _ in 0..8 {
                let row: Vec<u8> = (0..len)
                    .map(|_| match rng.next() % 3 {
                        0 => 0,
                        1 => (rng.next() % (t as u32 + 2)) as u8,
                        _ => cap,
                    })
                    .collect();
                let mut scan = row.clone();
                kernels::horizontal_min_plus_scalar(&mut scan);
                let mut window = vec![0u8; len];
                kernels::horizontal_window_min_row(&row, t, &mut window);
                let sel = |v: &[u8]| v.iter().map(|&d| usize::from(d) <= t).collect::<Vec<_>>();
                assert_eq!(sel(&window), sel(&scan), "t {t}, row {row:?}");
            }
        }
    }
}

#[test]
fn u8_and_u16_distance_lanes_agree() {
    // Every threshold whose cap fits a byte runs on u8 lanes; the u16 lanes
    // must select exactly the same pixels, including at the largest u8
    // threshold (cap 255) and on the parallel path.
    for (w, h) in [(41, 29), (300, 230)] {
        let src = labels_u16(w, h, 11);
        for connectivity in CONNECTIVITIES {
            for threshold in [0u16, 1, 4, 37, 253, 254] {
                let mut a = vec![0u16; w * h];
                let mut b = vec![0u16; w * h];
                kernels::erode_label_boundaries_with::<u16, u8>(
                    &src,
                    &mut a,
                    w,
                    h,
                    threshold,
                    connectivity,
                    9,
                );
                kernels::erode_label_boundaries_with::<u16, u16>(
                    &src,
                    &mut b,
                    w,
                    h,
                    threshold,
                    connectivity,
                    9,
                );
                assert_eq!(a, b, "{w}x{h} t={threshold} {connectivity:?}");
            }
        }
    }
}

#[test]
fn matches_oracle_u16_distance_lanes() -> TestResult {
    // A threshold above 254 needs u16 lanes: a thin image keeps the
    // O(radius^2) oracle cheap.
    let (w, h) = (270, 2);
    for connectivity in CONNECTIVITIES {
        for radius in [256, 262] {
            let mut data = vec![3u8; w * h];
            data[5] = 4;
            data[w + 200] = 5;
            let src = Image::<u8, 1>::new(sz(w, h), data.clone())?;
            let threshold = distance_threshold(w, h, radius, connectivity)?;
            assert!(threshold.is_some_and(|t| t >= u16::from(u8::MAX)));
            let mut dst = Image::<u8, 1>::from_size_val(src.size(), 0)?;
            erode_label_boundaries(&src, &mut dst, radius, connectivity, 0)?;
            let expected = oracle(&data, w, h, radius, connectivity, 0);
            assert_eq!(dst.as_slice(), &expected[..], "r={radius} {connectivity:?}");
        }
    }
    Ok(())
}

#[test]
fn size_mismatch_errors() -> TestResult {
    let src = Image::<u8, 1>::from_size_val(sz(4, 4), 1)?;
    let mut dst = Image::<u8, 1>::from_size_val(sz(4, 5), 0)?;
    let err = erode_label_boundaries(&src, &mut dst, 1, Connectivity::Four, 0);
    assert!(matches!(err, Err(ImageError::InvalidImageSize(..))));
    // Same pixel count, different shape: still rejected, `dst` untouched.
    let mut dst = Image::<u8, 1>::from_size_val(sz(2, 8), 7)?;
    let err = erode_label_boundaries(&src, &mut dst, 1, Connectivity::Four, 0);
    assert!(matches!(err, Err(ImageError::InvalidImageSize(..))));
    assert!(dst.as_slice().iter().all(|&v| v == 7));
    Ok(())
}

#[test]
fn threshold_overflow_errors() -> TestResult {
    // A 65537-pixel row has in-image distances up to 65536, above the
    // supported 65534; a radius that large is rejected rather than wrong.
    let w = 65_537;
    let src = Image::<u8, 1>::from_size_val(sz(w, 1), 1)?;
    let mut dst = Image::<u8, 1>::from_size_val(sz(w, 1), 0)?;
    let err = erode_label_boundaries(&src, &mut dst, 70_000, Connectivity::Four, 0);
    assert!(matches!(
        err,
        Err(ImageError::RadiusTooLarge {
            radius: 70_000,
            max: 65_535
        })
    ));
    // The largest supported threshold still runs.
    erode_label_boundaries(&src, &mut dst, 65_535, Connectivity::Four, 0)?;
    assert_eq!(dst.as_slice(), src.as_slice());
    Ok(())
}

/// Parity with scikit-image 0.21: `find_boundaries(labels, connectivity,
/// mode="thick")`, dilated by a diamond / square of radius `radius - 1`, on
/// a fixture written by
/// `benches/bench_label_boundaries_baselines.py --write-fixture`.
///
/// `tests/data/label_boundaries_skimage.bin`, little endian: `u64` width,
/// `u64` height, the `u16` labels, then one `u8` mask (1 = eroded) per
/// (connectivity 1, 2) × radius (1, 3) pair, in that order.
#[test]
fn skimage_find_boundaries_parity_fixture() -> TestResult {
    const FILL: u16 = u16::MAX;
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../tests/data/label_boundaries_skimage.bin");
    let bytes = std::fs::read(&path)?;
    let word = |at: usize| -> Result<usize, Box<dyn std::error::Error>> {
        let b: [u8; 8] = bytes
            .get(at..at + 8)
            .ok_or("truncated fixture")?
            .try_into()?;
        Ok(usize::try_from(u64::from_le_bytes(b))?)
    };
    let (w, h) = (word(0)?, word(8)?);
    let n = w * h;
    let labels: Vec<u16> = bytes
        .get(16..16 + 2 * n)
        .ok_or("truncated fixture")?
        .chunks_exact(2)
        .map(|b| u16::from_le_bytes([b[0], b[1]]))
        .collect();
    let mut masks = bytes
        .get(16 + 2 * n..)
        .ok_or("truncated fixture")?
        .chunks_exact(n);
    assert_eq!(masks.len(), 4);
    let src = Image::<u16, 1>::new(sz(w, h), labels)?;
    let mut dst = Image::<u16, 1>::from_size_val(src.size(), 0)?;
    for connectivity in CONNECTIVITIES {
        for radius in [1, 3] {
            let want = masks.next().ok_or("missing mask")?;
            erode_label_boundaries(&src, &mut dst, radius, connectivity, FILL)?;
            for (i, (&got, &m)) in dst.as_slice().iter().zip(want).enumerate() {
                assert_eq!(got == FILL, m != 0, "{connectivity:?} r{radius} pixel {i}");
                if m == 0 {
                    assert_eq!(got, src.as_slice()[i]);
                }
            }
        }
    }
    Ok(())
}
