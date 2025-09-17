use kornia_image::Image;
use kornia_imgproc::features::HarrisResponse;
use kornia_tensor::CpuAllocator;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Synthetic 9x9 u8 image with a bright rectangle
    let src = Image::from_size_slice(
        [9, 9].into(),
        &[
            0u8, 0, 0, 0, 0, 0, 0, 0, 0,
            0u8, 0, 0, 0, 0, 0, 0, 0, 0,
            0u8, 0, 255, 255, 255, 255, 255, 0, 0,
            0u8, 0, 255, 255, 255, 255, 255, 0, 0,
            0u8, 0, 255, 255, 255, 255, 255, 0, 0,
            0u8, 0, 255, 255, 255, 255, 255, 0, 0,
            0u8, 0, 255, 255, 255, 255, 255, 0, 0,
            0u8, 0, 0, 0, 0, 0, 0, 0, 0,
            0u8, 0, 0, 0, 0, 0, 0, 0, 0,
        ],
        CpuAllocator,
    )?;

    let mut dst = Image::<f32, 1, _>::from_size_val(src.size(), 0.0, CpuAllocator)?;
    HarrisResponse::new(src.size()).compute_u8(&src, &mut dst)?;

    // Print center 5x5 block of responses
    for r in 2..7 {
        for c in 2..7 {
            let v = dst.as_slice()[r * dst.cols() + c];
            print!("{v:.6} ");
        }
        println!();
    }

    Ok(())
}


