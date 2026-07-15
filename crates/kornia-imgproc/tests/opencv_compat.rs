//! Validation of the OpenCV-compat resize against reference vectors generated
//! by `cv2.resize` itself (see `tests/data/opencv_resize/generate.py`).
//!
//! The comparison contract encodes an empirical finding: a shipped cv2 wheel
//! is NOT internally byte-consistent for `INTER_LINEAR`. The aarch64 wheel
//! that generated these fixtures routes different channel counts and scale
//! factors through different backends (OpenCV's own code, Carotene, Arm
//! KleidiCV), whose least-significant bits disagree with each other and with
//! OpenCV's reference scalar arithmetic in `resize.cpp` (u8 by up to 2 LSB on
//! ~3% of pixels, f32 by up to 3 ulp on ~18%); `cv2.setUseOptimized(False)`
//! does not bypass a compile-time HAL. `INTER_NEAREST` is pure index-gather
//! and is byte-exact on every route.
//!
//! We therefore implement OpenCV's reference scalar semantics exactly (pinned
//! byte-for-byte by the unit tests in `resize::opencv_compat`), and validate
//! against the cv2 fixtures with the tight corridor measured above: exact for
//! nearest, ≤2 LSB for u8 linear, ≤4 ulp for f32 linear.

use kornia_image::{Image, ImageSize};
use kornia_imgproc::interpolation::InterpolationMode;
use kornia_imgproc::resize::{resize_opencv_f32, resize_opencv_u8};

struct Case {
    key: String,
    dtype: String,
    channels: usize,
    src_h: usize,
    src_w: usize,
    dst_h: usize,
    dst_w: usize,
    interp: String,
}

/// Every parameter is encoded in the fixture filename —
/// `<dtype>_c<channels>_<srcH>x<srcW>_to_<dstH>x<dstW>_<interp>` — so the
/// test needs no manifest file and no JSON dependency.
fn parse_key(key: &str) -> Case {
    let parts: Vec<&str> = key.split('_').collect();
    assert_eq!(parts.len(), 6, "unexpected fixture name: {key}");
    let dims = |s: &str| -> (usize, usize) {
        let (a, b) = s.split_once('x').expect("HxW");
        (a.parse().unwrap(), b.parse().unwrap())
    };
    let (src_h, src_w) = dims(parts[2]);
    let (dst_h, dst_w) = dims(parts[4]);
    assert_eq!(parts[3], "to", "unexpected fixture name: {key}");
    Case {
        key: key.to_string(),
        dtype: parts[0].to_string(),
        channels: parts[1].strip_prefix('c').unwrap().parse().unwrap(),
        src_h,
        src_w,
        dst_h,
        dst_w,
        interp: parts[5].to_string(),
    }
}

fn data_dir() -> std::path::PathBuf {
    std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/data/opencv_resize")
}

fn run_case<const C: usize>(case: &Case) {
    let dir = data_dir();
    let src_bytes = std::fs::read(dir.join(format!("{}.src", case.key))).unwrap();
    let dst_bytes = std::fs::read(dir.join(format!("{}.dst", case.key))).unwrap();
    let mode = match case.interp.as_str() {
        "linear" => InterpolationMode::Bilinear,
        "nearest" => InterpolationMode::Nearest,
        other => panic!("unknown interp {other}"),
    };
    let src_size = ImageSize {
        width: case.src_w,
        height: case.src_h,
    };
    let dst_size = ImageSize {
        width: case.dst_w,
        height: case.dst_h,
    };

    match case.dtype.as_str() {
        "u8" => {
            let src = Image::<u8, C>::new(src_size, src_bytes).unwrap();
            let mut dst = Image::<u8, C>::from_size_val(dst_size, 0).unwrap();
            resize_opencv_u8(&src, &mut dst, mode).unwrap();
            let max_lsb: i32 = match mode {
                InterpolationMode::Nearest => 0,
                _ => 2, // vendor-HAL corridor, see module doc
            };
            for (i, (got, want)) in dst.as_slice().iter().zip(&dst_bytes).enumerate() {
                let d = (*got as i32 - *want as i32).abs();
                assert!(
                    d <= max_lsb,
                    "{}: u8 element {i}: got {got} cv2 {want} ({d} LSB)",
                    case.key
                );
            }
        }
        "f32" => {
            let to_f32 = |b: &[u8]| -> Vec<f32> {
                b.chunks_exact(4)
                    .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                    .collect()
            };
            let src = Image::<f32, C>::new(src_size, to_f32(&src_bytes)).unwrap();
            let expected = to_f32(&dst_bytes);
            let mut dst = Image::<f32, C>::from_size_val(dst_size, 0.0).unwrap();
            resize_opencv_f32(&src, &mut dst, mode).unwrap();
            let max_ulp: i64 = match mode {
                InterpolationMode::Nearest => 0,
                _ => 4, // vendor-HAL corridor, see module doc
            };
            for (i, (got, want)) in dst.as_slice().iter().zip(&expected).enumerate() {
                let d = (got.to_bits() as i64 - want.to_bits() as i64).abs();
                assert!(
                    d <= max_ulp,
                    "{}: f32 element {i}: got {got} ({:#010x}) cv2 {want} ({:#010x}), {d} ulp",
                    case.key,
                    got.to_bits(),
                    want.to_bits()
                );
            }
        }
        other => panic!("unknown dtype {other}"),
    }
}

#[test]
fn resize_matches_cv2_reference_vectors() {
    let mut keys: Vec<String> = std::fs::read_dir(data_dir())
        .unwrap()
        .filter_map(|e| {
            let name = e.unwrap().file_name().into_string().unwrap();
            name.strip_suffix(".dst").map(str::to_string)
        })
        .collect();
    keys.sort();
    assert!(!keys.is_empty(), "no cv2 fixtures found");
    println!("validating {} cv2-generated fixtures", keys.len());
    for key in &keys {
        let case = &parse_key(key);
        match case.channels {
            1 => run_case::<1>(case),
            3 => run_case::<3>(case),
            n => panic!("unexpected channel count {n}"),
        }
    }
}
