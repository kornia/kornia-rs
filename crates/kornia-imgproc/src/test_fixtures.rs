//! Test data of the OpenCV parity tests: the MH_01 frames and the outputs
//! `benches/opencv_kernels_fixtures.py` writes; its docstring has the layouts
//! and the OpenCV parameters these constants repeat.

use kornia_image::Image;
use std::path::PathBuf;

pub(crate) const MAX_CORNERS: usize = 200;
pub(crate) const QUALITY: f32 = 0.01;
pub(crate) const MIN_DISTANCE: f32 = 23.0;
pub(crate) const BORDER: usize = 9;
pub(crate) const SUBPIX_HALF_WIN: usize = 5;
pub(crate) const SUBPIX_MAX_ITER: usize = 30;
pub(crate) const EPSILON: f32 = 0.01;

/// Size of the remap fixture's output, half of `mh01_frame1.png`.
pub(crate) const REMAP_SIZE: [usize; 2] = [376, 240];

pub(crate) fn data_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../tests/data")
}

/// A grayscale PNG of `tests/data`.
pub(crate) fn frame(name: &str) -> Image<u8, 1> {
    let img = kornia_io::png::read_image_png_mono8(data_dir().join(name)).unwrap();
    Image::new(img.size(), img.as_slice().to_vec()).unwrap()
}

/// The OpenCV outputs on `mh01_frame1.png` / `mh01_frame2.png`.
pub(crate) struct KernelFixture {
    pub frame1: Image<u8, 1>,
    pub frame2: Image<u8, 1>,
    /// `goodFeaturesToTrack` corners.
    pub corners: Vec<[f32; 2]>,
    /// `cornerSubPix` of `corners`.
    pub refined: Vec<[f32; 2]>,
    /// `calcOpticalFlowPyrLK` of `refined` into `frame2`.
    pub tracked: Vec<[f32; 2]>,
    /// `calcOpticalFlowPyrLK` status.
    pub status: Vec<u8>,
}

/// Reads `tests/data/frontend_kernels_opencv.bin`; panics on a missing or
/// truncated file, as a test should.
pub(crate) fn kernels() -> KernelFixture {
    let b = std::fs::read(data_dir().join("frontend_kernels_opencv.bin")).unwrap();
    let u32_at = |o: usize| u32::from_le_bytes(b[o..o + 4].try_into().unwrap()) as usize;
    let f32_at = |o: usize| f32::from_le_bytes(b[o..o + 4].try_into().unwrap());
    let n = u32_at(0);
    let points = |o: usize| -> Vec<[f32; 2]> {
        (0..n)
            .map(|i| [f32_at(o + 8 * i), f32_at(o + 8 * i + 4)])
            .collect()
    };
    let status_at = 4 + 24 * n;
    assert_eq!(b.len(), status_at + n, "frontend_kernels_opencv.bin size");
    KernelFixture {
        frame1: frame("mh01_frame1.png"),
        frame2: frame("mh01_frame2.png"),
        corners: points(4),
        refined: points(4 + 8 * n),
        tracked: points(4 + 16 * n),
        status: b[status_at..].to_vec(),
    }
}

/// The remap maps of the fixture: `map_x = x·1.96 + y·0.031 + 3.3`,
/// `map_y = y·1.93 − x·0.017 + 5.7`, in f32, left to right, as the script.
pub(crate) fn remap_maps() -> (Image<f32, 1>, Image<f32, 1>) {
    let [w, h] = REMAP_SIZE;
    let size = kornia_image::ImageSize {
        width: w,
        height: h,
    };
    let at = |f: fn(f32, f32) -> f32| -> Image<f32, 1> {
        let v = (0..w * h)
            .map(|i| f((i % w) as f32, (i / w) as f32))
            .collect();
        Image::new(size, v).unwrap()
    };
    (
        at(|x, y| x * 1.96 + y * 0.031 + 3.3),
        at(|x, y| y * 1.93 - x * 0.017 + 5.7),
    )
}

/// Reads `tests/data/remap_q5_opencv.bin`: OpenCV's remap of
/// `mh01_frame1.png` through [`remap_maps`].
pub(crate) fn remap_q5() -> Vec<u8> {
    let b = std::fs::read(data_dir().join("remap_q5_opencv.bin")).unwrap();
    let u32_at = |o: usize| u32::from_le_bytes(b[o..o + 4].try_into().unwrap()) as usize;
    assert_eq!(
        [u32_at(0), u32_at(4)],
        REMAP_SIZE,
        "remap_q5_opencv.bin size"
    );
    let n = REMAP_SIZE[0] * REMAP_SIZE[1];
    assert_eq!(b.len(), 8 + n, "remap_q5_opencv.bin size");
    b[8..].to_vec()
}

#[test]
fn the_kernel_fixture_has_one_entry_per_corner() {
    let fx = kernels();
    assert!(!fx.corners.is_empty());
    assert_eq!(fx.frame1.size(), fx.frame2.size());
    for len in [fx.refined.len(), fx.tracked.len(), fx.status.len()] {
        assert_eq!(len, fx.corners.len());
    }
}
