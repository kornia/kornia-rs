use clap::Parser;
use kornia::core::{CpuAllocator, Tensor};
use kornia::imgproc::interpolation::InterpolationMode;
use std::path::PathBuf;

use kornia::image::Image;
use kornia::io::functional as F;

#[derive(Parser)]
struct Args {
    #[arg(short, long)]
    image_path: PathBuf,

    #[arg(short, long)]
    onnx_model_path: PathBuf,

    #[arg(long)]
    ort_dylib_path: PathBuf,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = Args::parse();

    // set the ort dylib path
    std::env::set_var("ORT_DYLIB_PATH", &args.ort_dylib_path);

    // read the image
    let image: Image<u8, 3> = F::read_image_any(&args.image_path)?;

    // read the onnx model

    use ort::{GraphOptimizationLevel, Session};

    let model = Session::builder()?
        .with_optimization_level(GraphOptimizationLevel::Level3)?
        .with_intra_threads(4)?
        .commit_from_file(&args.onnx_model_path)?;

    let new_size = [320, 320].into();

    let mut image_resized = Image::from_size_val(new_size, 0)?;
    kornia::imgproc::resize::resize_fast(&image, &mut image_resized, InterpolationMode::Bilinear)?;

    let image = image_resized.clone().cast_and_scale::<f32>(1. / 255.)?;

    let image_chw = image.permute_axes([2, 0, 1]);

    // make it contiguous
    let mut new_data = Vec::<f32>::with_capacity(3 * 320 * 320);

    // TODO: implement Tensor::contiguous in kornia-core
    for i in 0..image.rows() {
        for j in 0..image.cols() {
            for c in 0..image.num_channels() {
                let value = image_chw.get_unchecked([i, j, c]);
                new_data.push(*value);
            }
        }
    }
    let image_nchw = Tensor::from_shape_slice([1, 3, 320, 320], &new_data, CpuAllocator)?;

    let ort_tensor = ort::Tensor::from_array((image_nchw.shape, image_nchw.into_vec()))?;

    // run the model
    let outputs = model.run(ort::inputs![
        "images" => ort_tensor,
    ]?)?;

    // get the output
    let (loc_shape, loc) = outputs["loc"].try_extract_raw_tensor::<f32>()?;
    let (conf_shape, conf) = outputs["conf"].try_extract_raw_tensor::<f32>()?;
    let (iou_shape, iou) = outputs["iou"].try_extract_raw_tensor::<f32>()?;

    println!("loc: {:?}", loc);
    println!("conf: {:?}", conf);
    println!("iou: {:?}", iou);

    Ok(())
}
