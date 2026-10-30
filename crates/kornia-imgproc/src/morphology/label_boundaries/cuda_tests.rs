//! Host vs device parity: the CUDA kernels must equal the CPU exactly.

use super::*;
use crate::cuda::color::test_utils::default_stream;
use kornia_image::ImageSize;

type TestResult = Result<(), Box<dyn std::error::Error>>;

/// Includes a single pixel, a single row / column, widths around and above
/// one horizontal tile (640 elements), a height above one vertical segment,
/// and a map with interior fused tiles (32 x 64 plus halo, no clamping).
const SIZES: [(usize, usize); 9] = [
    (1, 1),
    (1, 9),
    (9, 1),
    (5, 4),
    (67, 43),
    (640, 3),
    (1283, 5),
    (131, 97),
    (131, 211),
];

/// Radii up to 9 (threshold 8) take the fused single-launch kernel, larger
/// ones the two-launch path; 9 and 10 sit on either side of the switch.
const RADII: [usize; 9] = [0, 1, 2, 3, 5, 9, 10, 17, 70];

const CONNECTIVITIES: [Connectivity; 2] = [Connectivity::Four, Connectivity::Eight];

fn sz(width: usize, height: usize) -> ImageSize {
    ImageSize { width, height }
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
}

/// Blocky regions with sparse speckle.
fn labels_u16(w: usize, h: usize, seed: u64) -> Vec<u16> {
    let mut rng = Lcg(seed);
    let mut out = Vec::with_capacity(w * h);
    for y in 0..h {
        for x in 0..w {
            let r = rng.next();
            let region = ((x / 9) ^ (y / 7)) % 5;
            out.push(if r.is_multiple_of(31) {
                (r >> 8) as u16 % 11
            } else {
                40_000 + region as u16
            });
        }
    }
    out
}

fn check<T>(data: Vec<T>, w: usize, h: usize, fill: T) -> TestResult
where
    T: LabelPixel
        + cudarc::driver::DeviceRepr
        + cudarc::driver::ValidAsZeroBits
        + Default
        + std::fmt::Debug,
{
    let stream = default_stream();
    let src = Image::<T, 1>::new(sz(w, h), data)?;
    let d_src = src.to_cuda(&stream)?;
    for radius in RADII {
        for connectivity in CONNECTIVITIES {
            let mut cpu = Image::<T, 1>::from_size_val(src.size(), fill)?;
            erode_label_boundaries(&src, &mut cpu, radius, connectivity, fill)?;
            let mut d_dst = Image::<T, 1>::zeros_cuda(src.size(), &stream)?;
            erode_label_boundaries(&d_src, &mut d_dst, radius, connectivity, fill)?;
            let gpu = d_dst.to_host_owned()?;
            assert_eq!(
                gpu.as_slice(),
                cpu.as_slice(),
                "{w}x{h} r={radius} {connectivity:?}"
            );
        }
    }
    Ok(())
}

#[test]
fn erode_label_boundaries_device_equals_host_u8() -> TestResult {
    for (i, &(w, h)) in SIZES.iter().enumerate() {
        let data = labels_u16(w, h, i as u64)
            .into_iter()
            .map(|v| (v % 251) as u8)
            .collect();
        check::<u8>(data, w, h, 255)?;
    }
    Ok(())
}

#[test]
fn erode_label_boundaries_device_equals_host_u16() -> TestResult {
    for (i, &(w, h)) in SIZES.iter().enumerate() {
        check::<u16>(labels_u16(w, h, 100 + i as u64), w, h, 0)?;
    }
    Ok(())
}

#[test]
fn erode_label_boundaries_mixed_residency_errors() -> TestResult {
    let stream = default_stream();
    let size = sz(8, 8);
    let src = Image::<u8, 1>::from_size_val(size, 1)?;
    let mut d_dst = Image::<u8, 1>::zeros_cuda(size, &stream)?;
    let err = erode_label_boundaries(&src, &mut d_dst, 1, Connectivity::Four, 0);
    assert!(
        matches!(err, Err(ImageError::MixedResidency)),
        "got {err:?}"
    );
    Ok(())
}
