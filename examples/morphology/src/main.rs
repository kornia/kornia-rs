use argh::FromArgs;
use std::path::PathBuf;

use kornia::{
    image::{Image, ImageSize},
    imgproc::{color, morphology, padding, threshold},
    io::functional as F,
    tensor::CpuAllocator,
};

#[derive(FromArgs)]
/// Apply morphological operations (dilate, erode, open, close) to an image
struct Args {
    /// path to an input image
    #[argh(option, short = 'i')]
    image_path: PathBuf,

    /// kernel size (default: 5)
    #[argh(option, short = 's', default = "5")]
    kernel_size: usize,

    /// kernel shape: box, cross, ellipse (default: box)
    #[argh(option, short = 'k', default = "String::from(\"box\")")]
    kernel_shape: String,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Args = argh::from_env();

    // read the image
    let gray = F::read_image_any_rgb8(args.image_path)?;
    let size = gray.size();
    let (width, height) = (size.width, size.height);

    let mut gray_single = Image::<u8, 1, CpuAllocator>::new(
        ImageSize { width, height },
        vec![0u8; width * height],
        CpuAllocator,
    )?;

    // rgb to grayscale
    color::gray_from_rgb_u8(&gray, &mut gray_single)?;

    // apply threshold to create binary image
    let mut binary = Image::<u8, 1, CpuAllocator>::new(
        ImageSize { width, height },
        vec![0u8; width * height],
        CpuAllocator,
    )?;
    threshold::threshold_binary(
        &gray_single,
        &mut binary,
        128u8,
        255u8,
        kornia::imgproc::parallel::ExecutionStrategy::Serial,
    )?;

    let kernel_shape = match args.kernel_shape.as_str() {
        "cross" => morphology::KernelShape::Cross {
            size: args.kernel_size,
        },
        "ellipse" => morphology::KernelShape::Ellipse {
            width: args.kernel_size,
            height: args.kernel_size,
        },
        _ => morphology::KernelShape::Box {
            size: args.kernel_size,
        },
    };
    let kernel = morphology::Kernel::new(kernel_shape);

    let mut dilated = Image::<u8, 1, CpuAllocator>::new(
        ImageSize { width, height },
        vec![0u8; width * height],
        CpuAllocator,
    )?;

    let mut eroded = Image::<u8, 1, CpuAllocator>::new(
        ImageSize { width, height },
        vec![0u8; width * height],
        CpuAllocator,
    )?;

    let mut opened = Image::<u8, 1, CpuAllocator>::new(
        ImageSize { width, height },
        vec![0u8; width * height],
        CpuAllocator,
    )?;

    let mut closed = Image::<u8, 1, CpuAllocator>::new(
        ImageSize { width, height },
        vec![0u8; width * height],
        CpuAllocator,
    )?;

    // apply all morphological operations
    morphology::dilate(
        &binary,
        &mut dilated,
        &kernel,
        padding::PaddingMode::Constant,
        [0u8],
    )?;

    morphology::erode(
        &binary,
        &mut eroded,
        &kernel,
        padding::PaddingMode::Constant,
        [255u8],
    )?;

    morphology::open(
        &binary,
        &mut opened,
        &kernel,
        padding::PaddingMode::Constant,
        [255u8],
    )?;

    morphology::close(
        &binary,
        &mut closed,
        &kernel,
        padding::PaddingMode::Constant,
        [0u8],
    )?;

    // create a Rerun recording stream
    let rec = rerun::RecordingStreamBuilder::new("Kornia Morphology").spawn()?;

    // convert all images to rgb
    let mut gray_rgb = Image::<u8, 3, CpuAllocator>::new(
        ImageSize { width, height },
        vec![0u8; width * height * 3],
        CpuAllocator,
    )?;
    color::rgb_from_gray(&gray_single, &mut gray_rgb)?;

    let mut binary_rgb = Image::<u8, 3, CpuAllocator>::new(
        ImageSize { width, height },
        vec![0u8; width * height * 3],
        CpuAllocator,
    )?;
    color::rgb_from_gray(&binary, &mut binary_rgb)?;

    let mut dilated_rgb = Image::<u8, 3, CpuAllocator>::new(
        ImageSize { width, height },
        vec![0u8; width * height * 3],
        CpuAllocator,
    )?;
    color::rgb_from_gray(&dilated, &mut dilated_rgb)?;

    let mut eroded_rgb = Image::<u8, 3, CpuAllocator>::new(
        ImageSize { width, height },
        vec![0u8; width * height * 3],
        CpuAllocator,
    )?;
    color::rgb_from_gray(&eroded, &mut eroded_rgb)?;

    let mut opened_rgb = Image::<u8, 3, CpuAllocator>::new(
        ImageSize { width, height },
        vec![0u8; width * height * 3],
        CpuAllocator,
    )?;
    color::rgb_from_gray(&opened, &mut opened_rgb)?;

    let mut closed_rgb = Image::<u8, 3, CpuAllocator>::new(
        ImageSize { width, height },
        vec![0u8; width * height * 3],
        CpuAllocator,
    )?;
    color::rgb_from_gray(&closed, &mut closed_rgb)?;

    // log all images
    rec.log(
        "original",
        &rerun::Image::from_elements(
            gray.as_slice(),
            (width as u32, height as u32).into(),
            rerun::ColorModel::RGB,
        ),
    )?;

    rec.log(
        "grayscale",
        &rerun::Image::from_elements(
            gray_rgb.as_slice(),
            (width as u32, height as u32).into(),
            rerun::ColorModel::RGB,
        ),
    )?;

    rec.log(
        "binary",
        &rerun::Image::from_elements(
            binary_rgb.as_slice(),
            (width as u32, height as u32).into(),
            rerun::ColorModel::RGB,
        ),
    )?;

    rec.log(
        "operations/dilate",
        &rerun::Image::from_elements(
            dilated_rgb.as_slice(),
            (width as u32, height as u32).into(),
            rerun::ColorModel::RGB,
        ),
    )?;

    rec.log(
        "operations/erode",
        &rerun::Image::from_elements(
            eroded_rgb.as_slice(),
            (width as u32, height as u32).into(),
            rerun::ColorModel::RGB,
        ),
    )?;

    rec.log(
        "operations/open",
        &rerun::Image::from_elements(
            opened_rgb.as_slice(),
            (width as u32, height as u32).into(),
            rerun::ColorModel::RGB,
        ),
    )?;

    rec.log(
        "operations/close",
        &rerun::Image::from_elements(
            closed_rgb.as_slice(),
            (width as u32, height as u32).into(),
            rerun::ColorModel::RGB,
        ),
    )?;

    Ok(())
}
