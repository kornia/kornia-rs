//! CUDA imgproc — load a real image from disk, convert RGB→Gray on device.
//!
//! Usage:
//!   cargo run --release                      # uses default test image
//!   cargo run --release -- /path/to/img.jpg  # custom image path
//!
//! Pipeline (all device buffers stay kornia `Tensor`s):
//!   load RGB8 (kornia-io) → to_cuda (H2D) → kernel → to_host (D2H) → write PNG
//!
//! Eased CUDA API used here (from `kornia_tensor`, behind the `cuda` feature):
//!   - `Tensor::to_cuda`        — upload a host tensor to device (no manual flatten/copy)
//!   - `zeros_cuda`             — allocate a zero-filled device output tensor
//!   - `CudaKernel::compile`    — NVRTC compile with the device arch auto-detected
//!   - `launch_builder().arg()` — chained kernel-arg builder + `launch_1d`
//!   - `as_cudaslice` / `as_cudaslice_mut` — borrow device buffers as kernel args

use cudarc::driver::CudaContext;
use kornia_image::color_spaces::Gray8;
use kornia_image::image::ImageSize;
use kornia_io::functional::read_image_any_rgb8;
use kornia_io::png::write_image_png_gray8;
use kornia_tensor::{zeros_cuda, CudaKernel, Tensor};

// BT.601 RGB→gray CUDA kernel (u8 interleaved RGB → u8 planar gray).
// `src` is the flat h*w*3 byte buffer; `dst` is the flat h*w gray buffer.
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
    // Resolve the input path (CLI arg, else a bundled test image).
    let path = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "/home/nvidia/kornia-rs/tests/data/dog.jpeg".to_owned());
    println!("Loading image: {path}");

    // Load an RGB8 image from disk via kornia-io. `Rgb8` derefs/owns an
    // `Image<u8, 3>`, which is a newtype over `Tensor3<u8, _>`.
    let rgb = read_image_any_rgb8(&path)?;
    let (h, w) = (rgb.rows(), rgb.cols());
    let npix = h * w;
    println!("Image size: {w}x{h} ({npix} pixels)");

    let ctx = CudaContext::new(0)?;
    let stream = ctx.default_stream();

    // ── Eased device path: everything stays a kornia Tensor ──────────────────
    // Upload the RGB image directly — `to_cuda` copies the contiguous
    // h*w*3 bytes to a device tensor (no manual host flatten / Vec copy).
    // `rgb.0` is the inner `Image<u8, 3>`, whose `to_cuda` yields a `Tensor`
    // (the `Rgb8::to_cuda` that shadows it returns a device-resident `Rgb8`).
    let dev_rgb: Tensor<u8, 3> = rgb.0.to_cuda(&stream)?;
    // Allocate the device output as a zero-filled device tensor (no raw CudaSlice).
    let mut dev_gray = zeros_cuda::<u8, 1>([npix], &stream)?;

    // Compile (device arch auto-detected) and launch via the chained builder.
    let kernel = CudaKernel::compile(&ctx, RGB_TO_GRAY_SRC, "rgb_to_gray")?;
    kernel
        .launch_builder(&stream)
        .arg(dev_rgb.as_cudaslice().unwrap())
        .arg(dev_gray.as_cudaslice_mut().unwrap())
        .arg(&(npix as i32))
        .launch_1d(npix as u32)?;
    println!("Kernel launched: rgb_to_gray ({npix} pixels)");

    // Download the result back to a host tensor (D2H + stream sync).
    let gray = dev_gray.to_host(&stream)?;

    // Verify: print the first pixels and a u64 checksum.
    let gray_slice = gray.as_slice();
    let checksum: u64 = gray_slice.iter().map(|&b| b as u64).sum();
    println!(
        "First 8 gray pixels: {:?}",
        &gray_slice[..8.min(gray_slice.len())]
    );
    println!("Gray checksum: {checksum}");

    // Wrap the gray bytes as a Gray8 image and write a PNG.
    let gray_img = Gray8::from_size_vec(
        ImageSize {
            width: w,
            height: h,
        },
        gray_slice.to_vec(),
    )?;
    let out_path = "/tmp/cuda_imgproc_output.png";
    write_image_png_gray8(out_path, &gray_img)?;
    println!("Saved grayscale output to: {out_path}");

    println!("\nDone.");
    Ok(())
}
