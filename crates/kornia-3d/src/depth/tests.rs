use super::*;
use kornia_algebra::Vec3F64;
use kornia_image::ImageSize;

type TestResult = Result<(), Box<dyn std::error::Error>>;

/// `meter` of millimetre depth.
const MM: f32 = 1000.0;

fn sz(width: usize, height: usize) -> ImageSize {
    ImageSize { width, height }
}

fn depth_image<T: DepthSample>(
    w: usize,
    h: usize,
    data: Vec<T>,
    meter: f32,
) -> Result<DepthImage<T>, DepthError> {
    DepthImage::new(Image::new(sz(w, h), data)?, meter)
}

/// The smallest `meter` `T` accepts: the accepted set is an interval up to
/// `f32::MAX` (both conditions are monotone in `meter`), so a bisection over
/// the bit patterns of positive floats finds its lower end.
fn smallest_meter<T: DepthSample>() -> f32 {
    let (mut lo, mut hi) = (0u32, f32::MAX.to_bits());
    assert!(!T::meter_is_valid(f32::from_bits(lo)) && T::meter_is_valid(f32::from_bits(hi)));
    while hi - lo > 1 {
        let mid = lo + (hi - lo) / 2;
        if T::meter_is_valid(f32::from_bits(mid)) {
            hi = mid;
        } else {
            lo = mid;
        }
    }
    f32::from_bits(hi)
}

fn pinhole(fx: f64, fy: f64, cx: f64, cy: f64) -> PinholeCamera {
    PinholeCamera {
        fx,
        fy,
        cx,
        cy,
        k1: 0.0,
        k2: 0.0,
        p1: 0.0,
        p2: 0.0,
    }
}

struct Lcg(u64);
impl Lcg {
    fn next(&mut self) -> u32 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (self.0 >> 33) as u32
    }
    /// Uniform in `[0, 1)`.
    fn unit(&mut self) -> f32 {
        (self.next() >> 8) as f32 / (1u32 << 23) as f32
    }
}

/// Millimetre depth in `500..12000` with ~20% missing samples.
fn random_depth_u16(w: usize, h: usize, seed: u64) -> Vec<u16> {
    let mut rng = Lcg(seed);
    (0..w * h)
        .map(|_| {
            if rng.next().is_multiple_of(5) {
                0
            } else {
                500 + (rng.next() % 11_500) as u16
            }
        })
        .collect()
}

/// Metric depth in `0.3..8.0` with zeros, NaNs and a few negatives.
fn random_depth_f32(w: usize, h: usize, seed: u64) -> Vec<f32> {
    let mut rng = Lcg(seed);
    (0..w * h)
        .map(|_| match rng.next() % 10 {
            0 => 0.0,
            1 => f32::NAN,
            2 => -1.0,
            _ => 0.3 + 7.7 * rng.unit(),
        })
        .collect()
}

// ── unproject ──────────────────────────────────────────────────────────────

/// Values from the Open3D / pinhole formula `((x - cx) z / fx, (y - cy) z / fy, z)`,
/// worked by hand with exactly representable inputs.
#[test]
fn unproject_pins_reference_values() -> TestResult {
    let camera = pinhole(4.0, 2.0, 1.0, 0.5);
    let depth = depth_image(3, 2, vec![2.0, 0.0, 4.0, 1.0, 8.0, f32::NAN], 1.0)?;
    let cloud = depth.to_pointcloud(&camera, None)?;
    assert_eq!(
        cloud.points(),
        &vec![
            [-0.5, -0.5, 2.0],  // (0, 0): (0 - 1) * 2 / 4, (0 - 0.5) * 2 / 2
            [1.0, -1.0, 4.0],   // (2, 0)
            [-0.25, 0.25, 1.0], // (0, 1)
            [0.0, 2.0, 8.0],    // (1, 1)
        ]
    );
    assert!(cloud.colors().is_none() && cloud.normals().is_none());
    Ok(())
}

#[test]
fn unproject_u16_matches_formula_and_index() -> TestResult {
    let (w, h) = (97, 61);
    let data = random_depth_u16(w, h, 7);
    let depth = depth_image(w, h, data.clone(), MM)?;
    let camera = pinhole(91.5, 88.25, 47.3, 29.9);
    let mut points = Vec::new();
    let mut index = Vec::new();
    depth.to_pointcloud_into(&camera, None, &mut points, Some(&mut index))?;

    let expected: Vec<usize> = (0..w * h).filter(|&i| data[i] != 0).collect();
    assert_eq!(points.len(), expected.len());
    assert_eq!(
        index.iter().map(|&i| i as usize).collect::<Vec<_>>(),
        expected
    );
    for (p, &i) in points.iter().zip(&index) {
        let (x, y) = ((i as usize % w) as f64, (i as usize / w) as f64);
        // `1 / meter` rounds to the old `0.001` multiplier, so the depth is
        // bit-identical to the `metres_per_unit = 0.001` form.
        assert_eq!(p[2], f64::from(f32::from(data[i as usize]) * 0.001f32));
        let z = f64::from(data[i as usize]) * 1e-3;
        let want = [
            (x - camera.cx) / camera.fx * z,
            (y - camera.cy) / camera.fy * z,
            z,
        ];
        for k in 0..3 {
            assert!(
                (p[k] - want[k]).abs() <= 1e-5 * z.max(1.0),
                "pixel {i} axis {k}: {} vs {}",
                p[k],
                want[k]
            );
        }
    }
    Ok(())
}

/// Unproject -> project returns each point to its source pixel and depth.
#[test]
fn unproject_project_round_trip() -> TestResult {
    let (w, h) = (64, 48);
    let data = random_depth_f32(w, h, 11);
    let depth = depth_image(w, h, data.clone(), 1.0)?;
    let camera = pinhole(60.0, 62.5, 31.7, 23.2);
    let mut points = Vec::new();
    let mut index = Vec::new();
    depth.to_pointcloud_into(&camera, None, &mut points, Some(&mut index))?;
    let n_valid = data.iter().filter(|&&d| d > 0.0 && d.is_finite()).count();
    assert_eq!(points.len(), n_valid);
    for (p, &i) in points.iter().zip(&index) {
        let uv = camera
            .project_to_pixel(&Vec3F64::new(p[0], p[1], p[2]), 0.0)
            .ok_or("point behind the camera")?;
        let (x, y) = ((i as usize % w) as f64, (i as usize / w) as f64);
        assert!((uv.x - x).abs() < 1e-3 && (uv.y - y).abs() < 1e-3);
        assert_eq!(p[2], f64::from(data[i as usize]));
    }
    Ok(())
}

#[test]
fn unproject_mask_skips_pixels() -> TestResult {
    let (w, h) = (40, 30);
    let data = random_depth_u16(w, h, 3);
    let mut rng = Lcg(5);
    let mask: Vec<u8> = (0..w * h).map(|_| (rng.next() % 2) as u8 * 255).collect();
    let depth = depth_image(w, h, data.clone(), MM)?;
    let mask_img = Image::<u8, 1>::new(sz(w, h), mask.clone())?;
    let camera = pinhole(30.0, 30.0, 20.0, 15.0);
    let mut points = Vec::new();
    let mut index = Vec::new();
    depth.to_pointcloud_into(&camera, Some(&mask_img), &mut points, Some(&mut index))?;
    let expected: Vec<u32> = (0..w * h)
        .filter(|&i| data[i] != 0 && mask[i] != 0)
        .map(|i| i as u32)
        .collect();
    assert_eq!(index, expected);
    assert_eq!(points.len(), expected.len());
    Ok(())
}

/// The banded parallel writer equals the sequential reference, on heights
/// that do and do not divide into the band count.
#[test]
fn unproject_parallel_matches_sequential() -> TestResult {
    for &(w, h) in &[(1, 1), (5, 1), (1, 70), (33, 64), (17, 129), (211, 333)] {
        let data = random_depth_f32(w, h, (w * h) as u64);
        let camera = pinhole(50.0, 51.0, w as f64 / 2.0, h as f64 / 2.0);
        let p = kernels::unproject_params(&camera, &depth_image(w, h, data.clone(), 1.0)?)?;
        let (mut par_pts, mut par_ix) = (Vec::new(), Vec::new());
        let (mut seq_pts, mut seq_ix) = (Vec::new(), Vec::new());
        kernels::unproject_into(&data, None, &p, &mut par_pts, Some(&mut par_ix));
        kernels::unproject_sequential(&data, None, &p, &mut seq_pts, Some(&mut seq_ix));
        assert_eq!(par_pts, seq_pts, "{w}x{h}");
        assert_eq!(par_ix, seq_ix, "{w}x{h}");
    }
    Ok(())
}

/// The u16 `!= 0` test of the parallel path keeps exactly the pixels the
/// metres test of the sequential path keeps, with and without a mask, from
/// the largest `meter` (a subnormal `1 / meter`) to the smallest accepted
/// one.
#[test]
fn unproject_u16_parallel_matches_sequential() -> TestResult {
    let (w, h) = (97, 71);
    let data = random_depth_u16(w, h, 5);
    let mask: Vec<u8> = (0..w * h).map(|i| u8::from(i % 7 != 0)).collect();
    let camera = pinhole(60.0, 61.0, w as f64 / 2.0, h as f64 / 2.0);
    for meter in [MM, f32::MAX, smallest_meter::<u16>()] {
        let p = kernels::unproject_params(&camera, &depth_image(w, h, data.clone(), meter)?)?;
        for m in [None, Some(mask.as_slice())] {
            let (mut par_pts, mut par_ix) = (Vec::new(), Vec::new());
            let (mut seq_pts, mut seq_ix) = (Vec::new(), Vec::new());
            kernels::unproject_into(&data, m, &p, &mut par_pts, Some(&mut par_ix));
            kernels::unproject_sequential(&data, m, &p, &mut seq_pts, Some(&mut seq_ix));
            assert!(!seq_pts.is_empty() && seq_pts.len() < w * h);
            assert_eq!(par_pts, seq_pts, "meter {meter}, mask {}", m.is_some());
            assert_eq!(par_ix, seq_ix, "meter {meter}, mask {}", m.is_some());
        }
    }
    Ok(())
}

/// At every accepted `meter`, a u16 sample is valid iff non-zero, and that
/// equals the metres-value test of the sequential path, from `f32::MAX`
/// (a subnormal `1 / meter`) down to the smallest accepted `meter`
/// (`65535 * (1 / meter)` still finite). Smaller `meter`s are rejected.
#[test]
fn u16_validity_is_nonzero_at_every_accepted_meter() {
    let smallest = smallest_meter::<u16>();
    let below = f32::from_bits(smallest.to_bits() - 1);
    let meters = [
        f32::MAX,
        1.0 / f32::MIN_POSITIVE,
        MM,
        1.0,
        smallest,
        below,
        f32::MIN_POSITIVE,
        f32::from_bits(1),
    ];
    for m in meters {
        if !u16::meter_is_valid(m) {
            assert!((f32::from(u16::MAX) * (1.0 / m)).is_infinite(), "{m}");
            continue;
        }
        let s = 1.0 / m;
        for v in 0..=u16::MAX {
            assert_eq!(v.is_valid_at(s), v != 0, "v {v}, meter {m}");
            assert_eq!(
                v.is_valid_at(s),
                kernels::is_valid_depth_f32(v.metres(s)),
                "v {v}, meter {m}"
            );
        }
    }
    assert!(u16::meter_is_valid(smallest) && !u16::meter_is_valid(below));
    assert!(f32::meter_is_valid(f32::MAX));
}

/// The accepted `meter`s are the intervals the `DepthImage::new` rustdoc
/// states: `f32` from `1 / f32::MAX` (about `2.94e-39`, where the
/// reciprocal stops overflowing), `u16` from about `1.926e-34` (where
/// `65535 * (1 / meter)` stops overflowing), both up to `f32::MAX`.
#[test]
fn accepted_meters_are_the_documented_intervals() {
    let f = smallest_meter::<f32>();
    assert!((1.0 / f).is_finite() && (1.0 / f32::from_bits(f.to_bits() - 1)).is_infinite());
    assert!((f / 2.94e-39 - 1.0).abs() < 1e-3, "{f:e}");
    let u = smallest_meter::<u16>();
    assert!((u / 1.926e-34 - 1.0).abs() < 1e-3, "{u:e}");
    // Inside each interval every bit pattern up to `f32::MAX` is accepted
    // (sampled), and the `meter` of the old `0.001` multiplier rounds back
    // to it.
    for bits in (f.to_bits()..=f32::MAX.to_bits()).step_by(9973) {
        assert!(f32::meter_is_valid(f32::from_bits(bits)));
    }
    for bits in (u.to_bits()..=f32::MAX.to_bits()).step_by(9973) {
        assert!(u16::meter_is_valid(f32::from_bits(bits)));
    }
    assert_eq!(1.0f32 / MM, 0.001f32);
}

/// The validity rule (same as `kornia_imgproc::depth`): u16 valid iff
/// non-zero; f32 valid iff finite and positive, so `±0`, negative values,
/// `±inf` and `NaN` are missing while the smallest subnormal is a
/// measurement. Pinned on the predicates and on `to_pointcloud_into`.
#[test]
fn validity_rule_u16_nonzero_f32_finite_positive() -> TestResult {
    let f32_cases = [
        (0.0, false),
        (-0.0, false),
        (-1.0, false),
        (-f32::MIN_POSITIVE, false),
        (f32::INFINITY, false),
        (f32::NEG_INFINITY, false),
        (f32::NAN, false),
        (-f32::NAN, false),
        (f32::from_bits(1), true),
        (1.0, true),
        (f32::MAX, true),
    ];
    for (d, want) in f32_cases {
        assert_eq!(kernels::is_valid_depth_f32(d), want, "{d}");
        assert_eq!(d.is_valid_at(1.0), want, "{d}");
    }
    for (d, want) in [(0u16, false), (1, true), (u16::MAX, true)] {
        assert_eq!(kernels::is_valid_depth_u16(d), want, "{d}");
    }

    let camera = pinhole(1.0, 1.0, 0.0, 0.0);
    let n = f32_cases.len();
    let data: Vec<f32> = f32_cases.iter().map(|c| c.0).collect();
    let depth = depth_image(n, 1, data.clone(), 1.0)?;
    let mut points = Vec::new();
    let mut index = Vec::new();
    depth.to_pointcloud_into(&camera, None, &mut points, Some(&mut index))?;
    let kept: Vec<u32> = (0..n as u32).filter(|&i| f32_cases[i as usize].1).collect();
    assert_eq!(index, kept);
    for (p, &i) in points.iter().zip(&index) {
        assert_eq!(p[2], f64::from(data[i as usize]));
    }

    let depth = depth_image(3, 1, vec![0, 1, u16::MAX], MM)?;
    depth.to_pointcloud_into(&camera, None, &mut points, Some(&mut index))?;
    assert_eq!(index, vec![1, 2]);
    Ok(())
}

/// `DepthImage::new` accepts and rejects `meter` values per dtype: finite,
/// positive, with a finite reciprocal, and only `u16` rejects `meter`s
/// whose `65535 · (1 / meter)` overflows. An accepted image unprojects.
#[test]
fn depth_image_accepts_finite_positive_meters() -> TestResult {
    let camera = pinhole(2.0, 2.0, 1.0, 1.0);
    let meters = [
        0.0,
        -0.0,
        -1.0,
        f32::NAN,
        f32::INFINITY,
        f32::NEG_INFINITY,
        f32::from_bits(1),
        f32::MIN_POSITIVE / 2.0,
        f32::MIN_POSITIVE,
        1e-35,
        1e-30,
        1.0,
        MM,
        1e34,
        f32::MAX,
    ];
    for m in meters {
        let finite_positive = m.is_finite() && m > 0.0 && (1.0 / m).is_finite();
        let src = Image::<u16, 1>::from_size_val(sz(3, 2), 1000)?;
        let d = DepthImage::new(src, m);
        let want = finite_positive && (65535.0 * (1.0 / m)).is_finite();
        assert_eq!(d.is_ok(), want, "u16 meter {m}");
        match d {
            Ok(d) => assert_eq!(d.to_pointcloud(&camera, None)?.len(), 6),
            Err(e) => assert!(matches!(e, DepthError::InvalidMeter(_)), "{e}"),
        }

        let d = DepthImage::<f32>::from_size_val(sz(3, 2), 1000.0, m);
        assert_eq!(d.is_ok(), finite_positive, "f32 meter {m}");
        if let Ok(d) = d {
            d.to_pointcloud(&camera, None)?;
        }
    }

    Ok(())
}

#[test]
fn unproject_into_reuses_buffers() -> TestResult {
    let (w, h) = (50, 40);
    let depth = depth_image(w, h, random_depth_u16(w, h, 9), MM)?;
    let camera = pinhole(40.0, 40.0, 25.0, 20.0);
    let mut points = vec![[9.0; 3]; 7];
    let mut index = vec![9u32; 3];
    depth.to_pointcloud_into(&camera, None, &mut points, Some(&mut index))?;
    let first = points.clone();
    let (cap, ptr) = (points.capacity(), points.as_ptr());
    depth.to_pointcloud_into(&camera, None, &mut points, Some(&mut index))?;
    assert_eq!(points, first);
    assert_eq!(index.len(), points.len());
    assert_eq!((points.capacity(), points.as_ptr()), (cap, ptr));
    // Without an index buffer the points are the same.
    let mut plain = Vec::new();
    depth.to_pointcloud_into(&camera, None, &mut plain, None)?;
    assert_eq!(plain, first);
    Ok(())
}

/// Little-endian reader over a fixture's bytes.
struct Fixture<'a>(&'a [u8]);
impl Fixture<'_> {
    fn take<const N: usize>(&mut self) -> Result<[u8; N], Box<dyn std::error::Error>> {
        let (head, rest) = self.0.split_at_checked(N).ok_or("fixture truncated")?;
        self.0 = rest;
        Ok(head.try_into()?)
    }
    fn f64(&mut self) -> Result<f64, Box<dyn std::error::Error>> {
        Ok(f64::from_le_bytes(self.take()?))
    }
}

/// Parity with Open3D 0.20's legacy `PointCloud.create_from_depth_image`
/// on a fixture written by `benches/bench_depth_baselines.py
/// --write-fixtures`: a 40×30 random `u16` image over the full range with
/// ~20% zeros, non-square focal lengths and an off-centre principal point.
///
/// `tests/data/unproject_depth_open3d.bin`, little endian: `u32` width and
/// height, `f64` fx, fy, cx, cy and `depth_scale`, the `u16` pixels, then a
/// `u64` count and Open3D's `x, y, z` `f64` triplets in row-major order.
/// Open3D's `depth_scale` is the `meter`; Open3D divides by it and kornia
/// multiplies by its `f32` reciprocal, with `f32` arithmetic, so each
/// coordinate may differ by a few `f32` ulps of the point's depth.
#[test]
fn open3d_unproject_parity_fixture() -> TestResult {
    const ULPS: f64 = 4.0;
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../tests/data/unproject_depth_open3d.bin");
    let bytes = std::fs::read(&path)?;
    let mut f = Fixture(&bytes);
    let w = usize::try_from(u32::from_le_bytes(f.take()?))?;
    let h = usize::try_from(u32::from_le_bytes(f.take()?))?;
    let camera = pinhole(f.f64()?, f.f64()?, f.f64()?, f.f64()?);
    let depth_scale = f.f64()?;
    let pixels = (0..w * h)
        .map(|_| Ok(u16::from_le_bytes(f.take()?)))
        .collect::<Result<Vec<_>, Box<dyn std::error::Error>>>()?;
    let n = usize::try_from(u64::from_le_bytes(f.take()?))?;
    let expected = (0..n)
        .map(|_| Ok([f.f64()?, f.f64()?, f.f64()?]))
        .collect::<Result<Vec<_>, Box<dyn std::error::Error>>>()?;
    assert!(f.0.is_empty(), "trailing fixture bytes");
    assert_eq!(n, pixels.iter().filter(|&&d| d != 0).count());

    let depth = depth_image(w, h, pixels, depth_scale as f32)?;
    let cloud = depth.to_pointcloud(&camera, None)?;
    assert_eq!(cloud.len(), expected.len());
    for (a, b) in cloud.points().iter().zip(&expected) {
        let tol = ULPS * f64::from(f32::EPSILON) * b[2].abs().max(1.0);
        for k in 0..3 {
            assert!((a[k] - b[k]).abs() <= tol, "{a:?} vs {b:?}");
        }
    }
    Ok(())
}

#[test]
fn unproject_empty_and_all_invalid() -> TestResult {
    let camera = pinhole(1.0, 1.0, 0.0, 0.0);
    let zeros = DepthImage::<u16>::from_size_val(sz(8, 4), 0, MM)?;
    assert!(zeros.to_pointcloud(&camera, None)?.is_empty());
    let empty = depth_image::<f32>(0, 3, vec![], 1.0)?;
    assert!(empty.to_pointcloud(&camera, None)?.is_empty());
    Ok(())
}

#[test]
fn unproject_rejects_bad_arguments() -> TestResult {
    let camera = pinhole(1.0, 1.0, 0.0, 0.0);
    let depth = DepthImage::<u16>::from_size_val(sz(4, 3), 1, MM)?;
    // Same pixel count, different shape.
    let mask = Image::<u8, 1>::from_size_val(sz(3, 4), 1)?;
    assert!(matches!(
        depth.to_pointcloud(&camera, Some(&mask)),
        Err(DepthError::Image(ImageError::InvalidImageSize(4, 3, 3, 4)))
    ));
    for meter in [0.0, -1.0, f32::NAN, f32::INFINITY] {
        assert!(matches!(
            DepthImage::new(depth.image().clone(), meter),
            Err(DepthError::InvalidMeter(_))
        ));
    }
    for bad in [
        pinhole(0.0, 1.0, 0.0, 0.0),
        pinhole(1.0, f64::NAN, 0.0, 0.0),
        pinhole(1.0, 1.0, f64::INFINITY, 0.0),
    ] {
        assert!(matches!(
            depth.to_pointcloud(&bad, None),
            Err(DepthError::InvalidIntrinsics { .. })
        ));
    }
    Ok(())
}
