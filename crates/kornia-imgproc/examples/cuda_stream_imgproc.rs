//! CUDA-stream imgproc — rgb→gray on a kornia `Tensor` that **owns** device memory.
//!
//! This example demonstrates the redesigned `MemoryResource` ownership model:
//!
//! ```text
//! Image(host) → CudaSlice (H2D) → from_cudaslice → Tensor(Device, OWNS it)
//!   → kernel reads data_ptr() → output Tensor(Device) → to_host → checksum → drop frees
//! ```
//!
//! Build and run:
//! ```bash
//! cargo run -p kornia-imgproc --example cuda_stream_imgproc --features cuda-example
//! ```

use cudarc::driver::{CudaContext, LaunchConfig, PushKernelArg};
use cudarc::nvrtc::{compile_ptx_with_opts, CompileOptions};
use kornia_tensor::cuda::CudaAllocator;
use kornia_tensor::{CpuAllocator, Tensor};

// BT.601 rgb→gray CUDA kernel (u8 interleaved → u8 planar).
const RGB_TO_GRAY_SRC: &str = r#"
extern "C" __global__ void rgb_to_gray(
    const unsigned char* __restrict__ src,
    unsigned char*       __restrict__ dst,
    int npix)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= npix) return;
    int base = idx * 3;
    float r = (float)src[base];
    float g = (float)src[base + 1];
    float b = (float)src[base + 2];
    dst[idx] = (unsigned char)(0.299f * r + 0.587f * g + 0.114f * b + 0.5f);
}
"#;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // ── Step 1: create a synthetic 128×128 RGB gradient image on the host ─────
    //
    // We fall back to a synthetic gradient instead of requiring kornia-io,
    // keeping the example self-contained and build-flag-free.
    let width: usize = 128;
    let height: usize = 128;
    let npix: usize = width * height;

    // Build a simple gradient: pixel (row, col) → (col, row, 128)
    let rgb_data: Vec<u8> = (0..height)
        .flat_map(|row| {
            (0..width).flat_map(move |col| {
                [col as u8, row as u8, 128u8]
            })
        })
        .collect();

    // Wrap in a kornia Tensor<u8,1,CpuAllocator> (host, contiguous).
    // This is the source image: shape [npix*3] (interleaved RGB bytes).
    let host_rgb = Tensor::<u8, 1, CpuAllocator>::from_shape_vec(
        [npix * 3],
        rgb_data,
        CpuAllocator,
    )?;

    println!(
        "Step 1: host RGB tensor — shape {:?}, first 9 bytes: {:?}",
        host_rgb.shape,
        &host_rgb.as_slice()[..9],
    );

    // ── Step 2: H2D — CudaSlice, then wrap as Tensor (ZERO COPY) ─────────────
    //
    // `stream.clone_htod` copies host bytes to a new device allocation and
    // returns a `CudaSlice<u8>`.  We immediately hand that slice to
    // `Tensor::from_cudaslice`, which MOVES the slice (no copy, no alloc)
    // into a `CudaResource<u8>` owned by the tensor.
    //
    // After this call:
    //   - The tensor is the **sole owner** of the device allocation.
    //   - Dropping the tensor calls `CudaSlice::drop` → cudarc frees device mem.

    let ctx = CudaContext::new(0)?;
    let stream = ctx.default_stream();

    let dev_rgb_slice = stream.clone_htod(host_rgb.as_slice())?;

    // ZERO-COPY WRAP: Tensor NOW OWNS the device memory.
    let dev_rgb_tensor =
        Tensor::<u8, 1, CudaAllocator>::from_cudaslice(dev_rgb_slice, [npix * 3], stream.clone());

    println!(
        "Step 2: device RGB tensor — shape {:?}, domain: {:?}",
        dev_rgb_tensor.shape,
        dev_rgb_tensor.storage.domain(),
    );

    // ── Step 3: safety demo — as_slice() must panic on a Device tensor ────────
    //
    // `as_slice()` checks the `MemoryDomain` and panics with
    // "non-host-accessible" for Device tensors.  We catch the panic to prove
    // the invariant is enforced without aborting the example.
    let panic_result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        // This call panics because `dev_rgb_tensor` lives on the device.
        let _ = dev_rgb_tensor.as_slice();
    }));
    assert!(panic_result.is_err(), "expected a panic from as_slice on device tensor");
    println!("Step 3: as_slice() correctly refused host access on a Device tensor");

    // ── Step 4: allocate output device buffer for grayscale result ────────────
    //
    // We allocate a raw `CudaSlice<u8>` for the kernel output.  After the
    // kernel we wrap it in a Tensor too (demonstrating both input AND output
    // are Tensor-owned device allocations).
    let mut dev_gray_slice = stream.alloc_zeros::<u8>(npix)?;

    // ── Step 5: compile + launch the rgb_to_gray kernel via nvrtc ─────────────

    let ptx = compile_ptx_with_opts(
        RGB_TO_GRAY_SRC,
        CompileOptions {
            arch: Some("compute_87"),
            ..Default::default()
        },
    )
    .map_err(|e| format!("nvrtc compile error: {e}"))?;

    let module = ctx.load_module(ptx)?;
    let func = module.load_function("rgb_to_gray")?;

    // Borrow the input CudaSlice from the tensor for the kernel launch.
    // The tensor is alive for the entire unsafe block, so the borrow is valid.
    let input_cuda = dev_rgb_tensor
        .as_cudaslice()
        .ok_or("dev_rgb_tensor is not backed by CudaResource<u8>")?;

    let npix_i32 = npix as i32;

    let threads_per_block = 256u32;
    let blocks = (npix as u32).div_ceil(threads_per_block);
    let cfg = LaunchConfig {
        block_dim: (threads_per_block, 1, 1),
        grid_dim: (blocks, 1, 1),
        shared_mem_bytes: 0,
    };

    let mut builder = stream.launch_builder(&func);
    builder.arg(input_cuda);
    builder.arg(&mut dev_gray_slice);
    builder.arg(&npix_i32);
    unsafe { builder.launch(cfg) }?;

    println!("Step 5: rgb_to_gray kernel launched (grid={blocks} blocks × {threads_per_block} threads)");

    // ── Step 6: wrap output slice in a Tensor, then D2H ──────────────────────
    //
    // `from_cudaslice` moves `dev_gray_slice` into a `CudaResource<u8>`,
    // making the output tensor the sole owner of the output device buffer.
    let dev_gray_tensor =
        Tensor::<u8, 1, CudaAllocator>::from_cudaslice(dev_gray_slice, [npix], stream.clone());

    println!(
        "Step 6a: output device tensor — shape {:?}, domain: {:?}",
        dev_gray_tensor.shape,
        dev_gray_tensor.storage.domain(),
    );

    // `to_host` does a D2H copy and synchronizes the stream.
    let host_gray = dev_gray_tensor.to_host(&stream)?;

    // ── Step 7: verify — print first pixels and u64 checksum ─────────────────

    let gray_slice = host_gray.as_slice();
    let checksum: u64 = gray_slice.iter().map(|&b| b as u64).sum();
    println!(
        "Step 7: first 8 gray pixels: {:?}",
        &gray_slice[..8.min(gray_slice.len())]
    );
    println!("Step 7: gray checksum: {checksum}");

    // Quick sanity: BT.601 of (col=0, row=0, 128) = round(0+0+14.592) = 15
    {
        let expected_px0 =
            (0.299_f32 * 0.0 + 0.587_f32 * 0.0 + 0.114_f32 * 128.0 + 0.5).floor() as u8;
        assert_eq!(
            gray_slice[0], expected_px0,
            "pixel[0] mismatch: got {} expected {}",
            gray_slice[0], expected_px0
        );
    }

    // ── Step 8: show ownership — drop tensors and observe dealloc ─────────────
    //
    // The Tensor's Drop chain is:
    //   Tensor → TensorStorage → Box<dyn MemoryResource> → CudaResource<u8>
    //   → CudaSlice<u8>::drop → cudarc frees device memory (exactly once)

    println!("Step 8: dropping device tensors — device memory will be freed via Tensor drop");
    drop(dev_rgb_tensor);
    drop(dev_gray_tensor);
    println!("Step 8: device memory freed via Tensor drop");

    println!("\nDone — Tensor-owned CUDA device memory lifecycle verified.");
    Ok(())
}
