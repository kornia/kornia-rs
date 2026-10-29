//! Host vs device parity: the CUDA kernels must be bit-identical to the CPU.

use super::*;
use crate::cuda::color::test_utils::default_stream;
use kornia_image::ImageSize;

type TestResult = Result<(), Box<dyn std::error::Error>>;

const SIZES: [(usize, usize); 6] = [(1, 1), (2, 9), (5, 4), (67, 43), (64, 48), (131, 97)];

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

/// Piecewise-planar depth with noise, outliers and ~10 % holes: exercises
/// both flag rules and the even-count median.
fn depth_u16(w: usize, h: usize, channels: usize, seed: u64) -> Vec<u16> {
    let mut rng = Lcg(seed);
    let mut out = Vec::with_capacity(w * h * channels);
    for y in 0..h {
        for x in 0..w {
            for _ in 0..channels {
                let r = rng.next();
                let base = if (x / 7 + y / 5) % 2 == 0 { 900 } else { 2400 };
                let v = match r % 10 {
                    0 => 0,
                    1 => (r >> 12) as u16,
                    _ => base + (x + 2 * y) as u16 + (r >> 28) as u16,
                };
                out.push(v);
            }
        }
    }
    out
}

fn depth_f32(w: usize, h: usize, channels: usize, seed: u64) -> Vec<f32> {
    depth_u16(w, h, channels, seed)
        .into_iter()
        .map(|v| {
            if v == 0 {
                f32::NAN
            } else {
                f32::from(v) * 1e-3
            }
        })
        .collect()
}

fn bits(v: &[f32]) -> Vec<u32> {
    v.iter().map(|x| x.to_bits()).collect()
}

/// Host vs device median, compared bitwise through `bits_of` (NaN payload
/// included for `f32`).
fn check_median<T, const C: usize>(
    data: Vec<T>,
    w: usize,
    h: usize,
    ksize: usize,
    bits_of: fn(&T) -> u32,
) -> TestResult
where
    T: DepthPixel + cudarc::driver::DeviceRepr + cudarc::driver::ValidAsZeroBits + Default,
{
    let stream = default_stream();
    let src = Image::<T, C>::new(sz(w, h), data)?;
    let mut cpu = Image::<T, C>::from_size_val(src.size(), T::default())?;
    median_blur_depth(&src, &mut cpu, ksize)?;
    let d_src = src.to_cuda(&stream)?;
    let mut d_dst = Image::<T, C>::zeros_cuda(src.size(), &stream)?;
    median_blur_depth(&d_src, &mut d_dst, ksize)?;
    let gpu = d_dst.to_host_owned()?;
    let gpu: Vec<u32> = gpu.as_slice().iter().map(bits_of).collect();
    let cpu: Vec<u32> = cpu.as_slice().iter().map(bits_of).collect();
    assert_eq!(gpu, cpu, "C{C} {w}x{h} k={ksize}");
    Ok(())
}

#[test]
fn median_blur_depth_device_equals_host_u16() -> TestResult {
    let bits_of: fn(&u16) -> u32 = |v| u32::from(*v);
    for (i, &(w, h)) in SIZES.iter().enumerate() {
        for ksize in [3, 5] {
            check_median::<u16, 1>(depth_u16(w, h, 1, i as u64), w, h, ksize, bits_of)?;
            let data = depth_u16(w, h, 3, 100 + i as u64);
            check_median::<u16, 3>(data, w, h, ksize, bits_of)?;
        }
    }
    Ok(())
}

#[test]
fn median_blur_depth_device_equals_host_f32() -> TestResult {
    let bits_of: fn(&f32) -> u32 = |v| v.to_bits();
    for (i, &(w, h)) in SIZES.iter().enumerate() {
        for ksize in [3, 5] {
            let data = depth_f32(w, h, 1, 300 + i as u64);
            check_median::<f32, 1>(data, w, h, ksize, bits_of)?;
            let data = depth_f32(w, h, 3, 200 + i as u64);
            check_median::<f32, 3>(data, w, h, ksize, bits_of)?;
        }
    }
    Ok(())
}

/// Holes only in the left quarter: warps right of it see fully valid
/// windows and take the plain-median network, warps left of it do not.
fn holes_left_quarter_u16(w: usize, h: usize, seed: u64) -> Vec<u16> {
    let mut data = depth_u16(w, h, 1, seed);
    for (i, v) in data.iter_mut().enumerate() {
        if i % w >= w / 4 {
            *v = (*v).max(1);
        }
    }
    data
}

#[test]
fn median_blur_depth_device_equals_host_dense_tiles() -> TestResult {
    let bits_u16: fn(&u16) -> u32 = |v| u32::from(*v);
    let bits_f32: fn(&f32) -> u32 = |v| v.to_bits();
    for (i, &(w, h)) in SIZES.iter().enumerate() {
        for ksize in [3, 5] {
            let data = holes_left_quarter_u16(w, h, 600 + i as u64);
            let dense: Vec<u16> = data.iter().map(|&v| v.max(1)).collect();
            check_median::<u16, 1>(data.clone(), w, h, ksize, bits_u16)?;
            check_median::<u16, 1>(dense.clone(), w, h, ksize, bits_u16)?;
            // f32: holes cycle through every invalid encoding (`±0.0`,
            // negative, `±inf`, NaN); the dense image stays fully valid so
            // the all-valid fast path runs.
            const HOLES: [f32; 6] = [0.0, -0.0, -1.0, f32::INFINITY, f32::NEG_INFINITY, f32::NAN];
            let holed: Vec<f32> = data
                .iter()
                .enumerate()
                .map(|(i, &v)| {
                    if v == 0 {
                        HOLES[i % HOLES.len()]
                    } else {
                        f32::from(v) * 1e-3
                    }
                })
                .collect();
            let dense_f32: Vec<f32> = dense.iter().map(|&v| f32::from(v) * 1e-3).collect();
            check_median::<f32, 1>(holed, w, h, ksize, bits_f32)?;
            check_median::<f32, 1>(dense_f32, w, h, ksize, bits_f32)?;
        }
        let data = holes_left_quarter_u16(w, h, 700 + i as u64);
        let src = Image::<u16, 1>::new(sz(w, h), data)?;
        let mut cpu = Image::<u8, 1>::from_size_val(src.size(), 0)?;
        flying_pixel_mask(&src, &mut cpu, 30.0, 0.01, 6)?;
        let stream = default_stream();
        let d_src = src.to_cuda(&stream)?;
        let mut d_mask = Image::<u8, 1>::zeros_cuda(src.size(), &stream)?;
        flying_pixel_mask(&d_src, &mut d_mask, 30.0, 0.01, 6)?;
        assert_eq!(
            d_mask.to_host_owned()?.as_slice(),
            cpu.as_slice(),
            "{w}x{h}"
        );
    }
    Ok(())
}

#[test]
fn flying_pixel_mask_device_equals_host() -> TestResult {
    let stream = default_stream();
    let params = [
        (30.0f32, 0.0f32, 0usize),
        (0.0, 0.02, 5),
        (45.0, 0.01, 9),
        (0.0, 0.0, 10),
    ];
    for (i, &(w, h)) in SIZES.iter().enumerate() {
        for &(abs_t, rel_t, min_valid) in &params {
            let src = Image::<u16, 1>::new(sz(w, h), depth_u16(w, h, 1, 400 + i as u64))?;
            let mut cpu = Image::<u8, 1>::from_size_val(src.size(), 0)?;
            flying_pixel_mask(&src, &mut cpu, abs_t, rel_t, min_valid)?;
            let d_src = src.to_cuda(&stream)?;
            let mut d_mask = Image::<u8, 1>::zeros_cuda(src.size(), &stream)?;
            flying_pixel_mask(&d_src, &mut d_mask, abs_t, rel_t, min_valid)?;
            assert_eq!(
                d_mask.to_host_owned()?.as_slice(),
                cpu.as_slice(),
                "u16 {w}x{h} {abs_t} {rel_t} {min_valid}"
            );

            let src = Image::<f32, 1>::new(sz(w, h), depth_f32(w, h, 1, 500 + i as u64))?;
            flying_pixel_mask(&src, &mut cpu, abs_t * 1e-3, rel_t, min_valid)?;
            let d_src = src.to_cuda(&stream)?;
            flying_pixel_mask(&d_src, &mut d_mask, abs_t * 1e-3, rel_t, min_valid)?;
            assert_eq!(
                d_mask.to_host_owned()?.as_slice(),
                cpu.as_slice(),
                "f32 {w}x{h} {abs_t} {rel_t} {min_valid}"
            );
        }
    }
    // The fixture must actually exercise both outcomes.
    let (w, h) = (131, 97);
    let src = Image::<u16, 1>::new(sz(w, h), depth_u16(w, h, 1, 7))?;
    let mut cpu = Image::<u8, 1>::from_size_val(src.size(), 0)?;
    flying_pixel_mask(&src, &mut cpu, 30.0, 0.0, 0)?;
    let n_flagged = cpu.as_slice().iter().filter(|&&v| v != 0).count();
    assert!(n_flagged > 0 && n_flagged < w * h / 2, "{n_flagged}");
    Ok(())
}

#[test]
fn invalidate_masked_depth_device_equals_host() -> TestResult {
    let stream = default_stream();
    let (w, h) = (67, 43);
    let src = Image::<f32, 1>::new(sz(w, h), depth_f32(w, h, 1, 9))?;
    let mut mask = Image::<u8, 1>::from_size_val(src.size(), 0)?;
    flying_pixel_mask(&src, &mut mask, 0.025, 0.0, 4)?;
    let mut cpu = Image::<f32, 1>::from_size_val(src.size(), 0.0)?;
    invalidate_masked_depth(&src, &mask, &mut cpu)?;

    let d_src = src.to_cuda(&stream)?;
    let d_mask = mask.to_cuda(&stream)?;
    let mut d_dst = Image::<f32, 1>::zeros_cuda(src.size(), &stream)?;
    invalidate_masked_depth(&d_src, &d_mask, &mut d_dst)?;
    assert_eq!(
        bits(d_dst.to_host_owned()?.as_slice()),
        bits(cpu.as_slice())
    );

    let src = Image::<u16, 1>::new(sz(w, h), depth_u16(w, h, 1, 10))?;
    let mut mask = Image::<u8, 1>::from_size_val(src.size(), 0)?;
    flying_pixel_mask(&src, &mut mask, 30.0, 0.0, 4)?;
    let mut cpu = Image::<u16, 1>::from_size_val(src.size(), 0)?;
    invalidate_masked_depth(&src, &mask, &mut cpu)?;
    let d_src = src.to_cuda(&stream)?;
    let d_mask = mask.to_cuda(&stream)?;
    let mut d_dst = Image::<u16, 1>::zeros_cuda(src.size(), &stream)?;
    invalidate_masked_depth(&d_src, &d_mask, &mut d_dst)?;
    assert_eq!(d_dst.to_host_owned()?.as_slice(), cpu.as_slice());

    // A host mask with a device pair is a residency error, not a transfer.
    assert!(matches!(
        invalidate_masked_depth(&d_src, &mask, &mut d_dst),
        Err(ImageError::MixedResidency)
    ));
    Ok(())
}

#[test]
fn empty_device_images_are_a_no_op() -> TestResult {
    let stream = default_stream();
    for size in [sz(0, 5), sz(5, 0)] {
        // A zero-byte device allocation may be refused by the driver; the
        // ops cannot be reached then, so only a successful allocation is
        // tested.
        let (Ok(d_src), Ok(mut d_dst), Ok(mut d_mask)) = (
            Image::<u16, 1>::zeros_cuda(size, &stream),
            Image::<u16, 1>::zeros_cuda(size, &stream),
            Image::<u8, 1>::zeros_cuda(size, &stream),
        ) else {
            continue;
        };
        for ksize in [3, 5] {
            median_blur_depth(&d_src, &mut d_dst, ksize)?;
        }
        flying_pixel_mask(&d_src, &mut d_mask, 1.0, 0.0, 0)?;
        invalidate_masked_depth(&d_src, &d_mask, &mut d_dst)?;
    }
    Ok(())
}

#[test]
fn flying_pixel_mask_device_rejects_bad_thresholds() -> TestResult {
    let stream = default_stream();
    let src = Image::<u16, 1>::from_size_val(sz(8, 4), 1000)?.to_cuda(&stream)?;
    let mut mask = Image::<u8, 1>::zeros_cuda(src.size(), &stream)?;
    for (abs_t, rel_t) in [(f32::NAN, 0.0), (0.0, -1.0)] {
        assert!(matches!(
            flying_pixel_mask(&src, &mut mask, abs_t, rel_t, 0),
            Err(ImageError::InvalidThreshold { .. })
        ));
    }
    Ok(())
}
