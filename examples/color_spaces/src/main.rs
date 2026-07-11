//! Type-safe color conversions with residency dispatch.
//!
//! `convert` runs on the CPU, or on the GPU with `--features cuda`, selected by
//! where the images live. The call site is identical either way.

use kornia_imgproc::color::{ConvertColor, Gray8};
use kornia_io::functional as F;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let rgb = F::read_image_any_rgb8("../../tests/data/dog.jpeg")?;

    // Host images -> CPU path.
    let mut gray = Gray8::from_size_val(rgb.size(), 0)?;
    rgb.convert(&mut gray)?;
    println!(
        "host: {}x{} rgb -> gray, first pixel {}",
        rgb.width(),
        rgb.height(),
        gray.as_slice()[0]
    );

    // The same convert() on device images runs the CUDA kernel.
    #[cfg(feature = "cuda")]
    {
        use cudarc::driver::CudaContext;

        let stream = CudaContext::new(0)?.default_stream();
        let rgb_gpu = rgb.to_cuda(&stream)?;
        let mut gray_gpu = Gray8::zeros_cuda(rgb.size(), &stream)?;
        rgb_gpu.convert(&mut gray_gpu)?;

        let gray_gpu_host = gray_gpu.to_host(&stream)?;
        assert_eq!(gray.as_slice(), gray_gpu_host.as_slice());
        println!("gpu:  same convert() on device, output matches the CPU result");
    }

    Ok(())
}
