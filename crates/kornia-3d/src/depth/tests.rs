use super::*;
use kornia_algebra::{Mat3F64, Vec3F64};
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
/// equals the metres-value test of the sequential path and the scatter,
/// from `f32::MAX`
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
/// measurement. Pinned on the predicates, on `to_pointcloud_into` and on
/// `register_into`.
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

    // Registration onto itself skips the same samples. The valid ones are
    // replaced by an ordinary depth so that each lands on its own pixel
    // (extreme values such as `f32::MAX` are valid but do not project).
    let reg: Vec<f32> = f32_cases
        .iter()
        .map(|&(d, valid)| if valid { 2.5 } else { d })
        .collect();
    let camera = pinhole(4.0, 4.0, n as f64 / 2.0, 0.0);
    let src = depth_image(n, 1, reg, 1.0)?;
    let mut dst = DepthImage::<f32>::from_size_val(sz(n, 1), 7.0, 1.0)?;
    src.register_into(&camera, &Pose3d::IDENTITY, &camera, 0, &mut dst)?;
    for (i, &o) in dst.image().as_slice().iter().enumerate() {
        let want = if f32_cases[i].1 { 2.5 } else { 0.0 };
        assert_eq!(o.to_bits(), f32::to_bits(want), "pixel {i}");
    }
    Ok(())
}

/// `DepthImage::new` accepts and rejects `meter` values per dtype: finite,
/// positive, with a finite reciprocal, and only `u16` rejects `meter`s
/// whose `65535 · (1 / meter)` overflows. An accepted image both unprojects
/// and registers (the `meter` is checked once, so both ops accept exactly
/// the same set).
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
            Ok(d) => {
                assert_eq!(d.to_pointcloud(&camera, None)?.len(), 6);
                d.register(&camera, &Pose3d::IDENTITY, &camera, 0, sz(3, 2))?;
            }
            Err(e) => assert!(matches!(e, DepthError::InvalidMeter(_)), "{e}"),
        }

        let d = DepthImage::<f32>::from_size_val(sz(3, 2), 1000.0, m);
        assert_eq!(d.is_ok(), finite_positive, "f32 meter {m}");
        if let Ok(d) = d {
            d.to_pointcloud(&camera, None)?;
            d.register(&camera, &Pose3d::IDENTITY, &camera, 0, sz(3, 2))?;
        }
    }

    // The largest `meter` (a subnormal `1 / meter`) is usable, not just
    // accepted: 1000 units at `1 / f32::MAX` m (about 2.9e-36 m, a normal
    // float) register onto themselves.
    let data = vec![1000u16, 0, 2000, 1500, 0, 1200];
    let src = depth_image(3, 2, data.clone(), f32::MAX)?;
    let s = src.metres_per_unit();
    assert!(s > 0.0 && s < f32::MIN_POSITIVE);
    let mut dst = DepthImage::<u16>::from_size_val(sz(3, 2), 9, f32::MAX)?;
    src.register_into(&camera, &Pose3d::IDENTITY, &camera, 0, &mut dst)?;
    assert_eq!(dst.image().as_slice(), &data[..]);
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

/// Parity with Open3D 0.20's tensor `PointCloud.create_from_depth_image`
/// followed by `project_to_depth_image` (a nearest-pixel z-buffer, the model
/// of `DepthImage::register_into` at radius 0) on a fixture written by
/// `benches/bench_depth_baselines.py --write-fixtures`: a 40×30 `u16` plane
/// with a near box (21 destination pixels resolve an occlusion) registered
/// into a 44×32 camera with other intrinsics across a small rotation and
/// translation.
///
/// `tests/data/register_depth_open3d.bin`, little endian: `u32` width and
/// height and `f64` fx, fy, cx, cy of the source, the same of the
/// destination, `f64` `depth_scale`, `f64` row-major `[R | t]` of
/// `dst_from_src`, the `u16` source pixels, then Open3D's `f32` destination
/// depth in units (`0` = empty). The generator zeroes the source pixels
/// whose pixel, rounding or z-buffer decision is within rounding of a tie,
/// so the coverage and the winning point must match exactly; Open3D keeps
/// the value unrounded and divides by `depth_scale`, so the `f32` output may
/// differ by a few ulps and the `u16` output is Open3D's rounded.
#[test]
fn open3d_register_parity_fixture() -> TestResult {
    const ULPS: f32 = 8.0;
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../tests/data/register_depth_open3d.bin");
    let bytes = std::fs::read(&path)?;
    let mut f = Fixture(&bytes);
    let grid = |f: &mut Fixture| -> Result<_, Box<dyn std::error::Error>> {
        let w = usize::try_from(u32::from_le_bytes(f.take()?))?;
        let h = usize::try_from(u32::from_le_bytes(f.take()?))?;
        Ok(((w, h), pinhole(f.f64()?, f.f64()?, f.f64()?, f.f64()?)))
    };
    let ((w, h), src_cam) = grid(&mut f)?;
    let ((dw, dh), dst_cam) = grid(&mut f)?;
    let depth_scale = f.f64()?;
    let rt = (0..12).map(|_| f.f64()).collect::<Result<Vec<_>, _>>()?;
    let pixels = (0..w * h)
        .map(|_| Ok(u16::from_le_bytes(f.take()?)))
        .collect::<Result<Vec<_>, Box<dyn std::error::Error>>>()?;
    let expected = (0..dw * dh)
        .map(|_| Ok(f32::from_le_bytes(f.take()?)))
        .collect::<Result<Vec<_>, Box<dyn std::error::Error>>>()?;
    assert!(f.0.is_empty(), "trailing fixture bytes");
    // Column-major for `Mat3F64`.
    let rotation = Mat3F64::from_cols_array(&[
        rt[0], rt[4], rt[8], rt[1], rt[5], rt[9], rt[2], rt[6], rt[10],
    ]);
    let pose = Pose3d::new(rotation, Vec3F64::new(rt[3], rt[7], rt[11]));
    // Open3D's `depth_scale` is the `meter`.
    let meter = depth_scale as f32;
    let covered = expected.iter().filter(|&&d| d != 0.0).count();
    assert!(covered > dw * dh / 2, "{covered} covered pixels");

    let src = depth_image(w, h, pixels.clone(), meter)?;
    let mut dst = DepthImage::<u16>::from_size_val(sz(dw, dh), 9, meter)?;
    src.register_into(&src_cam, &pose, &dst_cam, 0, &mut dst)?;
    for (i, (&o, &e)) in dst.image().as_slice().iter().zip(&expected).enumerate() {
        assert_eq!(f32::from(o), e.round(), "u16 pixel {i}: {o} vs Open3D {e}");
    }

    let src = depth_image(w, h, pixels.iter().map(|&d| f32::from(d)).collect(), meter)?;
    let mut dst = DepthImage::<f32>::from_size_val(sz(dw, dh), 9.0, meter)?;
    src.register_into(&src_cam, &pose, &dst_cam, 0, &mut dst)?;
    for (i, (&o, &e)) in dst.image().as_slice().iter().zip(&expected).enumerate() {
        assert_eq!(o == 0.0, e == 0.0, "f32 pixel {i}: {o} vs Open3D {e}");
        assert!(
            (o - e).abs() <= ULPS * f32::EPSILON * e,
            "f32 pixel {i}: {o} vs Open3D {e}"
        );
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

// ── register ───────────────────────────────────────────────────────────────

fn translation(x: f64, y: f64, z: f64) -> Pose3d {
    Pose3d::new(Mat3F64::IDENTITY, Vec3F64::new(x, y, z))
}

/// Identity transform, same camera, no splat: every valid sample lands on
/// its own pixel with its own value; invalid samples become `0`.
#[test]
fn register_identity_reproduces_input() -> TestResult {
    let (w, h) = (73, 41);
    let camera = pinhole(70.0, 69.0, 36.2, 20.7);

    let data = random_depth_u16(w, h, 21);
    let src = depth_image(w, h, data.clone(), MM)?;
    let mut dst = DepthImage::<u16>::from_size_val(sz(w, h), 7, MM)?;
    src.register_into(&camera, &Pose3d::IDENTITY, &camera, 0, &mut dst)?;
    assert_eq!(dst.image().as_slice(), &data[..]);
    // The allocating form gives the same image, in the source's `meter`.
    let alloc = src.register(&camera, &Pose3d::IDENTITY, &camera, 0, sz(w, h))?;
    assert_eq!(alloc.image().as_slice(), &data[..]);
    assert_eq!(alloc.meter(), MM);

    let data = random_depth_f32(w, h, 22);
    let src = depth_image(w, h, data.clone(), 1.0)?;
    let mut dst = DepthImage::<f32>::from_size_val(sz(w, h), 7.0, 1.0)?;
    src.register_into(&camera, &Pose3d::IDENTITY, &camera, 0, &mut dst)?;
    for (i, (&o, &d)) in dst.image().as_slice().iter().zip(&data).enumerate() {
        let want = if d > 0.0 && d.is_finite() { d } else { 0.0 };
        assert_eq!(o.to_bits(), want.to_bits(), "pixel {i}");
    }
    Ok(())
}

/// An `f32` millimetre image (`meter = 1000`) registered onto itself
/// returns `(d · s) / s` with `s = 1 / meter` in `f32` bit for bit, as
/// documented: the input itself wherever that round trip is exact, never
/// more than 1 ulp away. Every integer millimetre value `1..=65535` is
/// covered; writing `(d · s) · meter` instead fails on 25 times more of
/// them, and `(d · s) · (1 / s)` (the old reciprocal form; `1 / s` rounds
/// to `999.99994`, not `1000`) on several times more, so this pins the
/// division by the inverse `meter`.
#[test]
fn register_f32_mm_identity_divides_by_inverse_meter() -> TestResult {
    let (w, h) = (256, 256);
    let data: Vec<f32> = (0..w * h).map(|i| i as f32).collect();
    let camera = pinhole(200.0, 200.0, 127.5, 127.5);
    let src = depth_image(w, h, data.clone(), MM)?;
    let mut dst = DepthImage::<f32>::from_size_val(sz(w, h), 7.0, MM)?;
    src.register_into(&camera, &Pose3d::IDENTITY, &camera, 0, &mut dst)?;
    let s = 1.0 / MM;
    assert_eq!(s, 0.001f32);
    let (mut exact, mut by_meter, mut by_reciprocal) = (0usize, 0usize, 0usize);
    for (i, (&o, &d)) in dst.image().as_slice().iter().zip(&data).enumerate() {
        let want = if d > 0.0 { (d * s) / s } else { 0.0 };
        assert_eq!(o.to_bits(), want.to_bits(), "pixel {i}");
        assert!(
            o.to_bits().abs_diff(d.to_bits()) <= 1,
            "pixel {i}: {o} vs {d}"
        );
        exact += usize::from(o == d);
        by_meter += usize::from((d * s) * MM == d);
        by_reciprocal += usize::from((d * s) * (1.0 / s) == d);
    }
    // 1510 of the 65535 non-zero values are 1 ulp off with the division,
    // 38588 with the multiplication by `meter`, 11483 with the reciprocal
    // (the same counts come out of NumPy float32).
    assert_eq!(w * h - exact, 1510);
    assert_eq!(w * h - by_meter, 38588);
    assert_eq!(w * h - by_reciprocal, 11483);
    Ok(())
}

/// Each image carries its own `meter`: the destination is written in its
/// own units, `z_dst / (1 / dst.meter())`. Millimetre `u16` registered
/// onto itself at a quarter-millimetre `meter` is `4 d`; millimetre `f32`
/// into a metric `f32` image is `d · 0.001` (rounded once more).
#[test]
fn register_converts_to_dst_meter() -> TestResult {
    let (w, h) = (64, 32);
    let camera = pinhole(60.0, 60.0, 31.5, 15.5);
    let data: Vec<u16> = (0..w * h).map(|i| (i % 16000) as u16).collect();
    let src = depth_image(w, h, data.clone(), MM)?;
    let mut dst = DepthImage::<u16>::from_size_val(sz(w, h), 9, 4.0 * MM)?;
    src.register_into(&camera, &Pose3d::IDENTITY, &camera, 0, &mut dst)?;
    for (i, (&o, &d)) in dst.image().as_slice().iter().zip(&data).enumerate() {
        assert_eq!(u32::from(o), 4 * u32::from(d), "pixel {i}");
    }

    let src = depth_image(w, h, data.iter().map(|&d| f32::from(d)).collect(), MM)?;
    let mut dst = DepthImage::<f32>::from_size_val(sz(w, h), 9.0, 1.0)?;
    src.register_into(&camera, &Pose3d::IDENTITY, &camera, 0, &mut dst)?;
    for (i, (&o, &d)) in dst.image().as_slice().iter().zip(&data).enumerate() {
        let z = f32::from(d) * (1.0 / MM);
        assert_eq!(o.to_bits(), (z / 1.0).to_bits(), "pixel {i}");
    }
    Ok(())
}

/// A foreground strip moves further (parallax) than the background behind
/// it and lands on top of it: the nearer depth must win every shared pixel,
/// and the background it uncovers is a hole.
#[test]
fn register_occlusion_nearer_wins() -> TestResult {
    let (w, h) = (24, 3);
    // fx * tx / z: background (4 m) moves 1 px, foreground (1 m) 4 px.
    let camera = pinhole(100.0, 100.0, 12.0, 1.0);
    let mut data = vec![4000u16; w * h];
    for y in 0..h {
        for x in 10..=13 {
            data[y * w + x] = 1000;
        }
    }
    let src = depth_image(w, h, data, MM)?;
    let mut dst = DepthImage::<u16>::from_size_val(sz(w, h), 0, MM)?;
    src.register_into(&camera, &translation(0.04, 0.0, 0.0), &camera, 0, &mut dst)?;
    let mut row = vec![0u16; w];
    row[1..=10].fill(4000); // background src 0..=9
    row[14..=17].fill(1000); // foreground src 10..=13, over background src 14..=16
    row[18..].fill(4000); // background src 17..=22; src 23 leaves the image
    for y in 0..h {
        assert_eq!(
            &dst.image().as_slice()[y * w..(y + 1) * w],
            &row[..],
            "row {y}"
        );
    }

    // The same scene in f32 metres.
    let metres = src.image().as_slice().iter();
    let src_f = depth_image(
        w,
        h,
        metres.map(|&d| f32::from(d) * (1.0 / MM)).collect(),
        1.0,
    )?;
    let mut dst_f = DepthImage::<f32>::from_size_val(sz(w, h), 0.0, 1.0)?;
    src_f.register_into(
        &camera,
        &translation(0.04, 0.0, 0.0),
        &camera,
        0,
        &mut dst_f,
    )?;
    let dst_f = dst_f.image().as_slice();
    for (i, (&o, &r)) in dst_f.iter().zip(row.iter().cycle()).enumerate() {
        assert!(
            (o - f32::from(r) * (1.0 / MM)).abs() < 1e-6,
            "pixel {i}: {o}"
        );
    }
    Ok(())
}

/// A 2x upsample leaves three holes per source pixel without a splat; a
/// radius-1 splat fills the whole destination.
#[test]
fn register_splat_fills_2x_upsample() -> TestResult {
    let (w, h) = (8, 6);
    let src_cam = pinhole(10.0, 10.0, 3.5, 2.5);
    // u_dst = 2 x + 0.25, so source pixel x lands on destination pixel 2x.
    let dst_cam = pinhole(20.0, 20.0, 7.25, 5.25);
    let src = DepthImage::<u16>::from_size_val(sz(w, h), 1500, MM)?;
    let mut dst = DepthImage::<u16>::from_size_val(sz(2 * w, 2 * h), 0, MM)?;

    src.register_into(&src_cam, &Pose3d::IDENTITY, &dst_cam, 0, &mut dst)?;
    let d = dst.image().as_slice();
    assert_eq!(d.iter().filter(|&&v| v != 0).count(), w * h);
    for y in 0..2 * h {
        for x in 0..2 * w {
            let want = if x % 2 == 0 && y % 2 == 0 { 1500 } else { 0 };
            assert_eq!(d[y * 2 * w + x], want, "({x}, {y})");
        }
    }

    src.register_into(&src_cam, &Pose3d::IDENTITY, &dst_cam, 1, &mut dst)?;
    assert!(dst.image().as_slice().iter().all(|&v| v == 1500));
    Ok(())
}

/// Points behind the destination camera, or whose depth does not fit the
/// `u16` output, are dropped.
#[test]
fn register_drops_unrepresentable_points() -> TestResult {
    let camera = pinhole(10.0, 10.0, 1.5, 1.5);
    let src = DepthImage::<u16>::from_size_val(sz(4, 4), 1000, MM)?;
    let mut dst = DepthImage::<u16>::from_size_val(sz(4, 4), 9, MM)?;
    // 1 m in front of the source, 1 m behind the destination.
    src.register_into(&camera, &translation(0.0, 0.0, -2.0), &camera, 0, &mut dst)?;
    assert!(dst.image().as_slice().iter().all(|&v| v == 0));
    // 1000 mm + 65 m = 66000 mm > 65534.
    src.register_into(&camera, &translation(0.0, 0.0, 65.0), &camera, 0, &mut dst)?;
    assert!(dst.image().as_slice().iter().all(|&v| v == 0));
    // 1000 mm + 64 m = 65000 mm still fits (the image shrinks onto the centre).
    src.register_into(&camera, &translation(0.0, 0.0, 64.0), &camera, 0, &mut dst)?;
    assert!(dst.image().as_slice().contains(&65000));
    Ok(())
}

#[test]
fn register_rejects_bad_arguments() -> TestResult {
    let camera = pinhole(1.0, 1.0, 0.0, 0.0);
    let src = DepthImage::<u16>::from_size_val(sz(4, 3), 1, MM)?;
    let mut dst = DepthImage::<u16>::from_size_val(sz(4, 3), 0, MM)?;
    let id = Pose3d::IDENTITY;
    assert!(matches!(
        src.register_into(&camera, &id, &camera, MAX_SPLAT_RADIUS + 1, &mut dst),
        Err(DepthError::SplatRadiusTooLarge { .. })
    ));
    assert!(matches!(
        src.register(&camera, &id, &camera, MAX_SPLAT_RADIUS + 1, sz(4, 3)),
        Err(DepthError::SplatRadiusTooLarge { .. })
    ));
    assert!(matches!(
        DepthImage::<u16>::from_size_val(sz(4, 3), 0, 0.0),
        Err(DepthError::InvalidMeter(_))
    ));
    assert!(matches!(
        src.register_into(&camera, &id, &pinhole(0.0, 1.0, 0.0, 0.0), 0, &mut dst),
        Err(DepthError::InvalidIntrinsics { .. })
    ));
    assert!(matches!(
        src.register_into(&pinhole(0.0, 1.0, 0.0, 0.0), &id, &camera, 0, &mut dst),
        Err(DepthError::InvalidIntrinsics { .. })
    ));
    assert!(matches!(
        src.register_into(
            &camera,
            &translation(f64::NAN, 0.0, 0.0),
            &camera,
            0,
            &mut dst
        ),
        Err(DepthError::InvalidTransform)
    ));
    Ok(())
}

/// Straight-line oracle of `DepthImage::register_into` for `u16` at
/// `meter = 1000` for both images: one sequential pass,
/// one pixel at a time, writing the smallest depth into each splat cell.
fn register_u16_oracle(
    src: &[u16],
    (w, h): (usize, usize),
    src_cam: &PinholeCamera,
    pose: &Pose3d,
    dst_cam: &PinholeCamera,
    radius: usize,
    (dw, dh): (usize, usize),
) -> Vec<u16> {
    let (ifx, ify) = ((1.0 / src_cam.fx) as f32, (1.0 / src_cam.fy) as f32);
    let (scx, scy) = (src_cam.cx as f32, src_cam.cy as f32);
    let (fx, fy) = (dst_cam.fx as f32, dst_cam.fy as f32);
    let (dcx, dcy) = (dst_cam.cx as f32, dst_cam.cy as f32);
    let m = pose.rotation.to_cols_array();
    let r: [f32; 9] = std::array::from_fn(|k| m[3 * (k % 3) + k / 3] as f32);
    let t = [
        pose.translation.x as f32,
        pose.translation.y as f32,
        pose.translation.z as f32,
    ];
    let rad = radius as i64;
    let s = 1.0 / MM;
    let mut out = vec![u16::MAX; dw * dh];
    for y in 0..h {
        for x in 0..w {
            let z = f32::from(src[y * w + x]) * s;
            if z <= 0.0 {
                continue;
            }
            let px = ((x as f32 - scx) * ifx) * z;
            let py = ((y as f32 - scy) * ify) * z;
            let qx = r[0] * px + r[1] * py + r[2] * z + t[0];
            let qy = r[3] * px + r[4] * py + r[5] * z + t[1];
            let qz = r[6] * px + r[7] * py + r[8] * z + t[2];
            if qz <= 0.0 {
                continue;
            }
            let iz = 1.0 / qz;
            let u = (fx * qx * iz + dcx + 0.5).floor() as i64;
            let v = (fy * qy * iz + dcy + 0.5).floor() as i64;
            let q = (qz / s).round();
            if !(1.0..=65534.0).contains(&q) {
                continue;
            }
            for yy in (v - rad).max(0)..=(v + rad).min(dh as i64 - 1) {
                for xx in (u - rad).max(0)..=(u + rad).min(dw as i64 - 1) {
                    let cell = &mut out[yy as usize * dw + xx as usize];
                    *cell = (*cell).min(q as u16);
                }
            }
        }
    }
    out.iter_mut()
        .filter(|c| **c == u16::MAX)
        .for_each(|c| *c = 0);
    out
}

/// The banded scatter equals the sequential oracle when splats stay in
/// their band's window and when strong rotations (roll, pitch) and
/// near-field parallax push them out of it into the deferred path.
#[test]
fn register_matches_sequential_oracle() -> TestResult {
    let rot = |roll: f64, pitch: f64| {
        let (c, s) = (roll.cos(), roll.sin());
        let rz = Mat3F64::from_cols_array(&[c, s, 0.0, -s, c, 0.0, 0.0, 0.0, 1.0]);
        let (c, s) = (pitch.cos(), pitch.sin());
        let rx = Mat3F64::from_cols_array(&[1.0, 0.0, 0.0, 0.0, c, s, 0.0, -s, c]);
        Mat3F64::from(rz.0 * rx.0)
    };
    let poses = [
        Pose3d::new(rot(0.0, 0.0), Vec3F64::new(0.05, 0.0, 0.0)),
        Pose3d::new(rot(0.5, 0.0), Vec3F64::new(0.0, 0.0, 0.0)),
        Pose3d::new(rot(0.1, 0.25), Vec3F64::new(0.02, 0.3, 0.1)),
        Pose3d::new(rot(-0.3, -0.2), Vec3F64::new(0.0, -0.4, 0.0)),
    ];
    for &((w, h), (dw, dh)) in &[
        ((160, 130), (160, 130)),
        ((97, 211), (194, 422)),
        ((200, 70), (100, 35)),
    ] {
        let src_cam = pinhole(
            0.8 * w as f64,
            0.8 * w as f64,
            w as f64 / 2.0,
            h as f64 / 2.0,
        );
        let dst_cam = pinhole(
            0.8 * dw as f64,
            0.8 * dw as f64,
            dw as f64 / 2.0,
            dh as f64 / 2.0,
        );
        let data = random_depth_u16(w, h, (w * h) as u64);
        let src = depth_image(w, h, data.clone(), MM)?;
        for pose in &poses {
            for radius in [0, 1, 3] {
                let want =
                    register_u16_oracle(&data, (w, h), &src_cam, pose, &dst_cam, radius, (dw, dh));
                let mut dst = DepthImage::<u16>::from_size_val(sz(dw, dh), 1, MM)?;
                src.register_into(&src_cam, pose, &dst_cam, radius, &mut dst)?;
                let got = dst.image().as_slice();
                assert_eq!(got, &want[..], "{w}x{h}->{dw}x{dh} r={radius}");
                assert!(want.iter().any(|&v| v != 0));
            }
        }
    }
    Ok(())
}

/// The NEON projection equals the scalar one bit for bit, lane by lane:
/// keys and pixels for both dtypes' keys, on random poses, destination
/// `meter`s (`1`, `1000`, `7`, so the output division is exercised) and depths that
/// include missing, negative, infinite, behind-camera and out-of-image
/// points, and on batch lengths that are not multiples of 4. Only aarch64
/// has a SIMD projection; elsewhere both calls would run the scalar path.
#[cfg(target_arch = "aarch64")]
#[test]
fn register_projection_simd_matches_scalar() -> TestResult {
    let mut rng = Lcg(77);
    let mut splats = 0usize;
    for case in 0..40 {
        let (w, h) = (
            64 + (rng.next() % 300) as usize,
            2 + (rng.next() % 300) as usize,
        );
        let cam = pinhole(
            20.0 + 900.0 * f64::from(rng.unit()),
            20.0 + 900.0 * f64::from(rng.unit()),
            w as f64 * f64::from(rng.unit()),
            h as f64 * f64::from(rng.unit()),
        );
        let (a, b) = (f64::from(rng.unit()) - 0.5, f64::from(rng.unit()) - 0.5);
        let rot = Mat3F64::from_cols_array(&[
            a.cos(),
            0.0,
            -a.sin(),
            a.sin() * b.sin(),
            b.cos(),
            a.cos() * b.sin(),
            a.sin() * b.cos(),
            -b.sin(),
            a.cos() * b.cos(),
        ]);
        let pose = Pose3d::new(
            rot,
            Vec3F64::new(
                f64::from(rng.unit()) - 0.5,
                f64::from(rng.unit()) - 0.5,
                2.0 * f64::from(rng.unit()) - 1.0,
            ),
        );
        let radius = (rng.next() % 4) as usize;
        let src = DepthImage::<f32>::from_size_val(sz(w, h), 0.0, 1.0)?;
        let dst_meter = [1.0, MM, 7.0][case % 3];
        let dst = DepthImage::<f32>::from_size_val(sz(w, h), 0.0, dst_meter)?;
        let p = kernels::register_params(&src, &cam, &pose, &cam, radius, &dst)?;
        let row: Vec<f32> = (0..kernels::PROJECT_BATCH)
            .map(|_| match rng.next() % 12 {
                0 => 0.0,
                1 => f32::NAN,
                2 => -1.0,
                3 => f32::INFINITY,
                4 => 1e-30,
                5 => 70_000.0,
                _ => 0.05 + 20.0 * rng.unit(),
            })
            .collect();
        let len = 1 + (rng.next() as usize % kernels::PROJECT_BATCH);
        let y = rng.next() as usize % h;
        for u16_keys in [false, true] {
            let mut simd = kernels::Batch::new();
            let mut scalar = kernels::Batch::new();
            kernels::project_batch(&row[..len], 3, y, &p, u16_keys, true, &mut simd);
            kernels::project_batch(&row[..len], 3, y, &p, u16_keys, false, &mut scalar);
            for j in 0..len {
                assert_eq!(simd.key[j], scalar.key[j], "case {case} lane {j}");
                if scalar.key[j] != 0 {
                    assert_eq!(
                        (simd.u[j], simd.v[j]),
                        (scalar.u[j], scalar.v[j]),
                        "case {case} lane {j}"
                    );
                }
            }
            splats += scalar.key[..len].iter().filter(|&&k| k != 0).count();
        }
    }
    // Not vacuous: plenty of lanes are real splats.
    assert!(splats > 200, "{splats} splats");
    Ok(())
}
