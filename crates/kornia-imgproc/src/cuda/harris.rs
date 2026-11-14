//! Fused f32 Harris response without shared memory, compiled through NVRTC.

use std::collections::HashMap;
use std::sync::{Arc, Mutex, OnceLock};

use cudarc::driver::{CudaContext, CudaSlice, CudaStream};
use kornia_tensor::CudaKernel;

super::define_cuda_error!(
    /// Error type for the CUDA Harris launcher.
    CudaHarrisError,
    "CUDA Harris error: {0}"
);

static HARRIS_SRC: &str = r#"
// Reference: features/responses.rs, HarrisResponse::compute (f32).
extern "C" __global__ void harris_fused_f32(
    const float* src, float* dst, int width, int height, float k)
{
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x < 1 || y < 1 || x >= width - 1 || y >= height - 1)
        return;

    float a = 0.0f, b = 0.0f, c = 0.0f;
    for (int dy = -1; dy <= 1; ++dy) {
        for (int dx = -1; dx <= 1; ++dx) {
            const int nx = x + dx, ny = y + dy;
            // The CPU leaves gradient products on the image border at zero.
            if (nx < 1 || ny < 1 || nx >= width - 1 || ny >= height - 1)
                continue;

            const int p = ny * width + nx;
            const float v11 = src[p - width - 1];
            const float v12 = src[p - width];
            const float v13 = src[p - width + 1];
            const float v21 = src[p - 1];
            const float v23 = src[p + 1];
            const float v31 = src[p + width - 1];
            const float v32 = src[p + width];
            const float v33 = src[p + width + 1];
            const float ix =
                (-v33 + v31 - 2.0f * v23 + 2.0f * v21 - v13 + v11) * 0.125f;
            const float iy =
                (-v33 - 2.0f * v32 - v31 + v13 + 2.0f * v12 + v11) * 0.125f;
            a += ix * ix;
            b += iy * iy;
            c += ix * iy;
        }
    }

    const float det = a * b - c * c;
    const float trace = a + b;
    dst[y * width + x] = fmaxf(0.0f, det - k * trace * trace);
}
"#;

type KernelMap = Mutex<HashMap<usize, Arc<CudaKernel>>>;
static KERNELS: OnceLock<KernelMap> = OnceLock::new();

fn get_kernel(ctx: &Arc<CudaContext>) -> Result<Arc<CudaKernel>, CudaHarrisError> {
    let cache = KERNELS.get_or_init(Default::default);
    let key = ctx.cu_ctx() as usize;
    let lock_error = |_| CudaHarrisError::Cuda("Harris kernel cache mutex poisoned".into());
    if let Some(kernel) = cache.lock().map_err(lock_error)?.get(&key).cloned() {
        return Ok(kernel);
    }
    let kernel = Arc::new(
        super::try_compile_with_l1(ctx, HARRIS_SRC, "harris_fused_f32")
            .map_err(CudaHarrisError::Cuda)?,
    );
    Ok(cache
        .lock()
        .map_err(lock_error)?
        .entry(key)
        .or_insert(kernel)
        .clone())
}

/// Launch the fused Harris kernel on device-resident f32 images.
///
/// Matches [`crate::features::HarrisResponse`] (Harris and Stephens, "A Combined
/// Corner and Edge Detector", 1988): 3x3 Sobel gradients scaled by 1/8, summed
/// over a 3x3 window.
///
/// # Arguments
///
/// * `ctx`, `stream` - Owning context and execution stream.
/// * `src`, `dst` - Row-major single-channel buffers; output borders are preserved.
/// * `width`, `height` - Image dimensions, each at least two pixels.
/// * `k` - Harris sensitivity, usually 0.04.
///
/// # Returns
///
/// `Ok(())` after enqueueing the kernel.
///
/// # Errors
///
/// Returns an error for invalid dimensions, undersized buffers, mismatched
/// contexts or CUDA failures.
///
/// # Example
///
/// ```no_run
/// use kornia_imgproc::cuda::harris::launch_harris_f32;
/// let ctx = cudarc::driver::CudaContext::new(0)?;
/// let stream = ctx.default_stream();
/// let src = stream.alloc_zeros::<f32>(25)?;
/// let mut dst = stream.alloc_zeros::<f32>(25)?;
/// launch_harris_f32(&ctx, &stream, &src, &mut dst, 5, 5, 0.04)?;
/// stream.synchronize()?;
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
#[allow(clippy::too_many_arguments)]
pub fn launch_harris_f32(
    ctx: &Arc<CudaContext>,
    stream: &Arc<CudaStream>,
    src: &CudaSlice<f32>,
    dst: &mut CudaSlice<f32>,
    width: usize,
    height: usize,
    k: f32,
) -> Result<(), CudaHarrisError> {
    let count = width
        .checked_mul(height)
        .filter(|&n| width >= 2 && height >= 2 && n <= i32::MAX as usize)
        .ok_or_else(|| {
            CudaHarrisError::Cuda("dimensions must be >= 2 and fit i32 indexing".into())
        })?;
    CudaHarrisError::check_slice("src", src.len(), count)?;
    CudaHarrisError::check_slice("dst", dst.len(), count)?;
    if ctx != stream.context() || ctx != src.stream().context() || ctx != dst.stream().context() {
        return Err(CudaHarrisError::Cuda(
            "buffers and stream must belong to ctx".into(),
        ));
    }
    let w = width as i32;
    let h = height as i32;
    let kernel = get_kernel(ctx)?;
    kernel
        .launch_builder(stream)
        .arg(src)
        .arg(dst)
        .arg(&w)
        .arg(&h)
        .arg(&k)
        .launch_cfg(super::make_config(w as u32, h as u32, None))
        .map_err(|e| CudaHarrisError::Cuda(e.to_string()))
}

#[cfg(all(test, feature = "cuda"))]
mod tests {
    use super::*;
    use crate::cuda::color::test_utils::{default_stream, pattern_f32};
    use crate::features::HarrisResponse;
    use kornia_image::{Image, ImageSize};

    #[test]
    fn harris_f32_close_to_cpu() {
        let stream = default_stream();
        for (width, height) in [(2, 2), (2, 9), (9, 2), (3, 3), (17, 33), (65, 23)] {
            let size = ImageSize { width, height };
            for source in [
                vec![0.0; width * height],
                vec![1.0; width * height],
                pattern_f32(width * height),
            ] {
                let src = Image::<f32, 1>::new(size, source).unwrap();
                let d_src = stream.clone_htod(src.as_slice()).unwrap();
                for k in [0.0, 0.04, 0.1] {
                    let mut cpu = Image::<f32, 1>::from_size_val(size, -7.0).unwrap();
                    HarrisResponse::new(size)
                        .with_k(k)
                        .compute(&src, &mut cpu)
                        .unwrap();
                    let mut d_dst = stream.clone_htod(&vec![-7.0f32; width * height]).unwrap();
                    launch_harris_f32(
                        stream.context(),
                        &stream,
                        &d_src,
                        &mut d_dst,
                        width,
                        height,
                        k,
                    )
                    .unwrap();
                    let gpu = stream.clone_dtoh(&d_dst).unwrap();
                    stream.synchronize().unwrap();
                    for (p, (&actual, &expected)) in gpu.iter().zip(cpu.as_slice()).enumerate() {
                        assert!(
                            (actual - expected).abs() <= 1e-6 + 1e-4 * expected.abs(),
                            "{width}x{height}, k={k}, pixel {p}: gpu={actual}, cpu={expected}"
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn harris_rejects_invalid_inputs() {
        let stream = default_stream();
        let src = stream.alloc_zeros::<f32>(4).unwrap();
        let mut dst = stream.alloc_zeros::<f32>(4).unwrap();
        for (w, h) in [
            (0, 2),
            (1, 2),
            (2, 1),
            (usize::MAX, 2),
            (i32::MAX as usize, 2),
            (3, 3),
        ] {
            assert!(
                launch_harris_f32(stream.context(), &stream, &src, &mut dst, w, h, 0.04).is_err()
            );
        }
        let src = stream.alloc_zeros::<f32>(9).unwrap();
        assert!(launch_harris_f32(stream.context(), &stream, &src, &mut dst, 3, 3, 0.04).is_err());
    }

    #[test]
    fn harris_available_devices() {
        for ordinal in 0..CudaContext::device_count().unwrap() as usize {
            let ctx = CudaContext::new(ordinal).unwrap();
            let stream = ctx.default_stream();
            let src = stream.alloc_zeros::<f32>(9).unwrap();
            let mut dst = stream.alloc_zeros::<f32>(9).unwrap();
            launch_harris_f32(&ctx, &stream, &src, &mut dst, 3, 3, 0.04).unwrap();
            stream.synchronize().unwrap();
            assert_eq!(stream.clone_dtoh(&dst).unwrap(), vec![0.0; 9]);
        }
    }

    #[test]
    fn harris_separate_contexts() {
        use cudarc::driver::sys;

        let primary = CudaContext::new(0).unwrap();
        let device = cudarc::driver::result::device::get(0).unwrap();
        let mut raw = std::ptr::null_mut();
        // SAFETY: the driver symbol has this ABI; device is valid and ownership transfers below.
        let separate = unsafe {
            // Resolve the legacy entry directly: cudarc's safe constructor omits CUDA 13.2.
            let create = sys::culib().get::<unsafe extern "C" fn(*mut sys::CUcontext, u32, sys::CUdevice) -> sys::CUresult>(b"cuCtxCreate_v2\0").unwrap();
            create(&mut raw, 0, device).result().unwrap();
            CudaContext::from_raw_context(0, device, raw).unwrap()
        };
        let first = get_kernel(&primary).unwrap();
        let second = get_kernel(&separate).unwrap();
        assert!(!Arc::ptr_eq(&first, &second));
        assert!(Arc::ptr_eq(&first, &get_kernel(&primary).unwrap()));
        for ctx in [&primary, &separate, &primary] {
            let stream = ctx.default_stream();
            let src = stream.alloc_zeros::<f32>(9).unwrap();
            let mut dst = stream.clone_htod(&[-7.0f32; 9]).unwrap();
            launch_harris_f32(ctx, &stream, &src, &mut dst, 3, 3, 0.04).unwrap();
            stream.synchronize().unwrap();
            let gpu = stream.clone_dtoh(&dst).unwrap();
            assert_eq!(gpu[4], 0.0);
            assert_eq!(gpu[0], -7.0);
        }
        let stream = primary.default_stream();
        let src = stream.alloc_zeros::<f32>(9).unwrap();
        let mut dst = stream.alloc_zeros::<f32>(9).unwrap();
        assert!(launch_harris_f32(&separate, &stream, &src, &mut dst, 3, 3, 0.04).is_err());
    }

    #[test]
    fn harris_worked_example() {
        let stream = default_stream();
        #[rustfmt::skip]
        let source = [
            0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0.,
            0., 0., 1., 1., 1.,
            0., 0., 1., 1., 1.,
            0., 0., 1., 1., 1.,
        ];
        let d_src = stream.clone_htod(&source).unwrap();
        let mut d_dst = stream.alloc_zeros::<f32>(source.len()).unwrap();
        launch_harris_f32(stream.context(), &stream, &d_src, &mut d_dst, 5, 5, 0.04).unwrap();
        let gpu = stream.clone_dtoh(&d_dst).unwrap();
        stream.synchronize().unwrap();
        assert!((gpu[12] - 0.49203125).abs() <= 1e-6);
    }
}
