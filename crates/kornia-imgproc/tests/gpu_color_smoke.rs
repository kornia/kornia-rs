#![cfg(feature = "gpu-cubecl")]

use cubecl::Runtime;
use cubecl_cuda::CudaRuntime;
use kornia_imgproc::gpu::color::launch_gray_from_rgb_f32;

/// Run with: cargo test -p kornia-imgproc --features gpu-cubecl -- --ignored
#[test]
#[ignore = "requires a CUDA device"]
fn gray_from_rgb_kernel_launches_on_cuda() {
    let device = <CudaRuntime as Runtime>::Device::default();
    let client = CudaRuntime::client(&device);

    let width: u32 = 4;
    let height: u32 = 4;
    let num_pixels = (width * height) as usize;

    // Allocate device buffers (content is uninitialized — this test only checks launch).
    let src = client.empty(num_pixels * 3 * std::mem::size_of::<f32>());
    let dst = client.empty(num_pixels * std::mem::size_of::<f32>());

    // Should not panic or return an error.
    launch_gray_from_rgb_f32::<CudaRuntime>(&client, src, dst, width, height);

    // Flush the queue — panics on driver error.
    client.sync().expect("kernel execution failed");
}
