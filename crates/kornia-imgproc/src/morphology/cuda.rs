//! Device adapter for [`dilate`](super::dilate) / [`erode`](super::erode):
//! routes u8 device pairs to the CUDA morphology kernel.
//!
//! The CPU ops are generic over `T: Ord`; the GPU kernel is u8-only. The
//! prologue in `ops.rs` forwards ANY device-touching call here, and this
//! adapter type-checks: u8 runs the kernel, every other element type gets
//! the typed no-GPU-kernel error (never a silent host fallback, which would
//! read device pointers on the CPU).

use std::any::TypeId;
use std::collections::HashMap;
use std::sync::{Arc, Mutex, OnceLock};

use cudarc::driver::CudaSlice;
use kornia_image::{Image, ImageError};

use crate::cuda::dispatch::{device_slices, dims_u32, no_gpu_kernel_err, untyped_device_err};
use crate::cuda::morphology::{launch_morphology_u8_cuda, MorphBorder, MorphOp};
use crate::padding::PaddingMode;

use super::kernels::Kernel;

/// Device upload of the per-channel border constant, cached per (device,
/// value, channels): Jetson pageable H2D uploads have a ~250 µs average
/// latency tail, and a warm cache keeps frame loops CUDA-Graph-capturable.
/// (The structuring element itself is baked into the kernel source and
/// cached by `cuda::morphology`.)
#[derive(PartialEq, Eq, Hash, Clone, Copy)]
struct CvalKey {
    dev: usize,
    cval: [u8; 4],
    channels: usize,
}

static CVAL_CACHE: OnceLock<Mutex<HashMap<CvalKey, Arc<CudaSlice<u8>>>>> = OnceLock::new();
const CVAL_CACHE_CAP: usize = 64;

fn map_border(mode: PaddingMode) -> MorphBorder {
    match mode {
        PaddingMode::Constant => MorphBorder::Constant,
        PaddingMode::Replicate => MorphBorder::Replicate,
        PaddingMode::Reflect101 => MorphBorder::Reflect101,
        PaddingMode::Reflect => MorphBorder::Reflect,
        PaddingMode::Wrap => MorphBorder::Wrap,
    }
}

/// Run a device morphology op. `T` is checked at runtime: only u8 has a GPU
/// kernel. Called by the `ops.rs` prologues whenever either operand is
/// device-resident (mixed pairs get the typed error from `pair_residency`).
pub(super) fn morphology_device<T: 'static, const C: usize>(
    src: &Image<T, C>,
    dst: &mut Image<T, C>,
    kernel: &Kernel,
    padding_mode: PaddingMode,
    constant_value: [T; C],
    op: MorphOp,
) -> Result<(), ImageError> {
    let op_name = match op {
        MorphOp::Dilate => "dilate",
        MorphOp::Erode => "erode",
    };
    if TypeId::of::<T>() != TypeId::of::<u8>() {
        return Err(no_gpu_kernel_err(op_name, "u8 device images"));
    }
    if !(C == 1 || C == 3 || C == 4) {
        return Err(no_gpu_kernel_err(op_name, "1/3/4-channel u8 images"));
    }
    if src.size() != dst.size() {
        return Err(ImageError::InvalidImageSize(
            dst.width(),
            dst.height(),
            src.width(),
            src.height(),
        ));
    }
    // SAFETY: TypeId proved T == u8 and `Image<T, C>` is a repr-transparent
    // newtype over a C-independent `Tensor3<T>`; reinterpreting the element
    // type to itself is a no-op cast.
    let src8 = unsafe { &*(src as *const Image<T, C> as *const Image<u8, C>) };
    let dst8 = unsafe { &mut *(dst as *mut Image<T, C> as *mut Image<u8, C>) };
    let cval_u8: [u8; C] = unsafe { *(std::ptr::from_ref(&constant_value) as *const [u8; C]) };

    let exec = match crate::cuda::dispatch::pair_residency(src8, dst8)? {
        crate::cuda::dispatch::Residency::Device(exec) => exec,
        // The ops.rs prologue only calls in when something is device-resident,
        // so a Host classification cannot happen; keep it as a typed error
        // rather than an unreachable! for robustness.
        crate::cuda::dispatch::Residency::Host => {
            return Err(ImageError::Cuda(
                "morphology_device called with host images".into(),
            ))
        }
    };

    exec.run(|stream| {
        let (w, h) = dims_u32(src8)?;
        let ctx = stream.context();
        let (s, d) = device_slices!(src8, dst8);

        // Active taps as (dy, dx) relative offsets — the same k_data == 1
        // set the CPU samples, with the padded-coordinate shift removed.
        // They are baked into the kernel source (compiled + cached per
        // element by the launcher), so no device LUT is uploaded.
        let (pad_h, pad_w) = kernel.pad();
        let (kw, kh) = (kernel.width(), kernel.height());
        let mut taps = Vec::<i32>::new();
        for ky in 0..kh {
            for kx in 0..kw {
                if kernel.data()[ky * kw + kx] == 1 {
                    taps.push(ky as i32 - pad_h as i32);
                    taps.push(kx as i32 - pad_w as i32);
                }
            }
        }

        let mut cval4 = [0u8; 4];
        cval4[..C].copy_from_slice(&cval_u8);
        let key = CvalKey {
            dev: ctx.ordinal(),
            cval: cval4,
            channels: C,
        };
        let cache = CVAL_CACHE.get_or_init(Default::default);
        // Scoped-guard lookup: binding `.get()` inside an `if let` scrutinee
        // would keep the MutexGuard alive through the else branch and
        // deadlock on the second `lock()`.
        let cached = cache
            .lock()
            .expect("cval cache poisoned")
            .get(&key)
            .cloned();
        let cval_dev = if let Some(hit) = cached {
            hit
        } else {
            let built = Arc::new(
                stream
                    .clone_htod(&cval_u8[..])
                    .map_err(|e| ImageError::Cuda(e.to_string()))?,
            );
            let mut map = cache.lock().expect("cval cache poisoned");
            if map.len() >= CVAL_CACHE_CAP {
                map.clear();
            }
            map.entry(key).or_insert(built).clone()
        };

        launch_morphology_u8_cuda(
            ctx,
            stream,
            s,
            d,
            w,
            h,
            C as u32,
            &taps,
            &cval_dev,
            op,
            map_border(padding_mode),
            None,
        )
        .map_err(|e| ImageError::Cuda(e.to_string()))
    })
}

// ── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use crate::cuda::color::test_utils::{default_stream, pattern_u8};
    use crate::morphology::{dilate, erode, Kernel, KernelShape};
    use crate::padding::PaddingMode;
    use kornia_image::{Image, ImageSize};

    /// The public `dilate`/`erode`, called with device images, must be
    /// bit-identical to the host path across structuring elements, border
    /// modes (including the constant fill), and channel counts.
    #[test]
    fn public_morphology_u8_device_equals_host() {
        let stream = default_stream();

        fn run<const C: usize>(
            w: usize,
            h: usize,
            kernel: &Kernel,
            mode: PaddingMode,
            cval: [u8; C],
            stream: &std::sync::Arc<cudarc::driver::CudaStream>,
        ) {
            let size = ImageSize {
                width: w,
                height: h,
            };
            let src = Image::<u8, C>::new(size, pattern_u8(w * h * C)).unwrap();
            let d_src = src.to_cuda(stream).unwrap();

            for op_is_dilate in [true, false] {
                let mut cpu_dst = Image::<u8, C>::from_size_val(size, 0).unwrap();
                let mut d_dst = Image::<u8, C>::zeros_cuda(size, stream).unwrap();
                if op_is_dilate {
                    dilate(&src, &mut cpu_dst, kernel, mode, cval).unwrap();
                    dilate(&d_src, &mut d_dst, kernel, mode, cval).unwrap();
                } else {
                    erode(&src, &mut cpu_dst, kernel, mode, cval).unwrap();
                    erode(&d_src, &mut d_dst, kernel, mode, cval).unwrap();
                }
                assert_eq!(
                    d_dst.to_host_owned().unwrap().as_slice(),
                    cpu_dst.as_slice(),
                    "{w}x{h} C={C} dilate={op_is_dilate} {mode:?}: device must equal host"
                );
            }
        }

        let box3 = Kernel::new(KernelShape::Box { size: 3 });
        let cross5 = Kernel::new(KernelShape::Cross { size: 5 });
        let ellipse = Kernel::new(KernelShape::Ellipse {
            width: 5,
            height: 3,
        });

        for mode in [
            PaddingMode::Constant,
            PaddingMode::Replicate,
            PaddingMode::Reflect101,
            PaddingMode::Reflect,
            PaddingMode::Wrap,
        ] {
            run::<1>(63, 41, &box3, mode, [7], &stream);
            run::<3>(63, 41, &cross5, mode, [7, 200, 128], &stream);
        }
        run::<4>(
            33,
            21,
            &ellipse,
            PaddingMode::Replicate,
            [0, 1, 2, 3],
            &stream,
        );
        // Even-sized box: the pad = size/2 asymmetric-offset case.
        let box4 = Kernel::new(KernelShape::Box { size: 4 });
        run::<3>(40, 30, &box4, PaddingMode::Reflect101, [0, 0, 0], &stream);
    }

    /// Non-u8 device pairs error with the typed no-GPU-kernel message
    /// instead of reading device memory on the host.
    #[test]
    fn morphology_non_u8_device_errors() {
        let stream = default_stream();
        let size = ImageSize {
            width: 16,
            height: 16,
        };
        let src = Image::<u16, 1>::new(size, vec![0u16; 16 * 16]).unwrap();
        let d_src = src.to_cuda(&stream).unwrap();
        let mut d_dst = Image::<u16, 1>::zeros_cuda(size, &stream).unwrap();
        let kernel = Kernel::new(KernelShape::Box { size: 3 });
        let err = dilate(&d_src, &mut d_dst, &kernel, PaddingMode::Replicate, [0]).unwrap_err();
        assert!(
            matches!(&err, kornia_image::ImageError::Cuda(m) if m.contains("u8")),
            "got {err:?}"
        );
    }
}

#[cfg(test)]
mod bench_probe {
    /// Quick device-time probe (not a benchmark harness): run with
    /// `cargo test ... morphology::bench_probe -- --ignored --nocapture`.
    #[test]
    #[ignore]
    fn probe_dilate_1080p() {
        use crate::cuda::color::test_utils::{default_stream, pattern_u8};
        use crate::morphology::{dilate, Kernel, KernelShape};
        use crate::padding::PaddingMode;
        use kornia_image::{Image, ImageSize};

        let stream = default_stream();
        let size = ImageSize {
            width: 1920,
            height: 1080,
        };
        let src = Image::<u8, 1>::new(size, pattern_u8(1920 * 1080)).unwrap();
        let d_src = src.to_cuda(&stream).unwrap();
        let mut d_dst = Image::<u8, 1>::zeros_cuda(size, &stream).unwrap();
        let se = Kernel::new(KernelShape::Box { size: 3 });

        for _ in 0..30 {
            dilate(&d_src, &mut d_dst, &se, PaddingMode::Replicate, [0]).unwrap();
        }
        stream.synchronize().unwrap();
        for round in 0..3 {
            let t0 = std::time::Instant::now();
            for _ in 0..300 {
                dilate(&d_src, &mut d_dst, &se, PaddingMode::Replicate, [0]).unwrap();
            }
            stream.synchronize().unwrap();
            println!(
                "dilate3x3 1080p C1 round {round}: {:.3} ms",
                t0.elapsed().as_secs_f64() * 1000.0 / 300.0
            );
        }
    }
}
