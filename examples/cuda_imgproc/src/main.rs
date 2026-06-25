//! CUDA imgproc — load a real image from disk, convert RGB→Gray on device.
//!
//! Usage:
//!   cargo run --release                      # uses default test image
//!   cargo run --release -- /path/to/img.jpg  # custom image path
//!
//! Pipeline:
//!   load RGB8 (kornia-io) → flatten Tensor<u8,1> → H2D → kernel → D2H → write PNG

use cudarc::driver::{CudaContext, CudaSlice};
use kornia_image::{allocator::CpuAllocator, color_spaces::Gray8, image::ImageSize};
use kornia_io::functional::read_image_any_rgb8;
use kornia_io::png::write_image_png_gray8;
use kornia_tensor::{CudaAllocator, CudaKernel, Tensor};

// BT.601 RGB→gray CUDA kernel (u8 interleaved → u8 planar).
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
    // ── Step 1: resolve image path ────────────────────────────────────────────
    let path = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "/home/nvidia/kornia-rs/tests/data/dog.jpeg".to_owned());

    println!("Loading image: {path}");

    // ── Step 2: load RGB8 from disk ───────────────────────────────────────────
    let rgb_img = read_image_any_rgb8(&path)?;
    let height = rgb_img.rows();
    let width = rgb_img.cols();
    let npix = height * width;

    println!("Image size: {width}x{height} ({npix} pixels)");

    // ── Step 3: flatten to Tensor<u8,1,CpuAllocator> ─────────────────────────
    // Image<u8,3,CpuAllocator> Derefs to Tensor3<u8,CpuAllocator>.
    // We borrow the raw bytes from as_slice() and construct a flat Tensor.
    let rgb_bytes: &[u8] = rgb_img.as_slice();
    let host_rgb =
        Tensor::<u8, 1, CpuAllocator>::from_shape_vec([npix * 3], rgb_bytes.to_vec(), CpuAllocator)?;

    println!(
        "Host RGB tensor — shape {:?}, first 9 bytes: {:?}",
        host_rgb.shape,
        &host_rgb.as_slice()[..9.min(host_rgb.as_slice().len())]
    );

    // ── Step 4: set up CUDA context + stream ─────────────────────────────────
    let ctx = CudaContext::new(0)?;
    let stream = ctx.default_stream();

    // ── Step 5: H2D — copy host RGB tensor to device ──────────────────────────
    let dev_rgb = host_rgb.to_cuda(&stream)?;

    println!(
        "Device RGB tensor — shape {:?}, domain: {:?}",
        dev_rgb.shape,
        dev_rgb.storage.domain(),
    );

    // ── Step 6: allocate output device buffer (raw CudaSlice) ────────────────
    let mut dev_gray_slice: CudaSlice<u8> = stream.alloc_zeros::<u8>(npix)?;

    // ── Step 7: compile + launch rgb_to_gray via CudaKernel ──────────────────
    let kernel = CudaKernel::compile(&ctx, RGB_TO_GRAY_SRC, "rgb_to_gray")?;

    let input_slice = dev_rgb
        .as_cudaslice()
        .ok_or("dev_rgb is not backed by CudaResource")?;

    let npix_i32 = npix as i32;

    kernel
        .launch_builder(&stream)
        .arg(input_slice)
        .arg(&mut dev_gray_slice)
        .arg(&npix_i32)
        .launch_1d(npix as u32)?;

    println!("Kernel launched: rgb_to_gray ({npix} pixels)");

    // ── Step 8: wrap output slice as Tensor, then D2H ────────────────────────
    let dev_gray_tensor =
        Tensor::<u8, 1, CudaAllocator>::from_cudaslice(dev_gray_slice, [npix], stream.clone());

    let host_gray_tensor = dev_gray_tensor.to_host(&stream)?;

    // ── Step 9: verify — print first pixels and u64 checksum ─────────────────
    let gray_slice = host_gray_tensor.as_slice();
    let checksum: u64 = gray_slice.iter().map(|&b| b as u64).sum();

    println!(
        "First 8 gray pixels: {:?}",
        &gray_slice[..8.min(gray_slice.len())]
    );
    println!("Gray checksum: {checksum}");

    // ── Step 10: wrap as Gray8 image and save PNG ─────────────────────────────
    let gray_img = Gray8::from_size_vec(
        ImageSize { width, height },
        gray_slice.to_vec(),
        CpuAllocator,
    )?;

    let out_path = "/tmp/cuda_imgproc_output.png";
    write_image_png_gray8(out_path, &gray_img)?;
    println!("Saved grayscale output to: {out_path}");

    println!("\nDone.");
    Ok(())
}
