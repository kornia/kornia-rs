use argh::FromArgs;
use kornia::image::ops;
use kornia::image::{Image, ImageSize};
use kornia::io::functional as F;
use kornia_imgproc::interpolation::InterpolationMode;
use kornia_imgproc::normalize::normalize_mean_std;
use kornia_imgproc::resize::resize_native;
use kornia_tensor::{CpuAllocator, Tensor};
use ort::session::builder::GraphOptimizationLevel;
use ort::session::Session;
use ort::value::Tensor as OrtTensor;
use std::path::{Path, PathBuf};
use std::time::Instant;

// SigLIP2 preprocessing parameters — matches kornia.models.siglip2.SigLip2ImagePreprocessor.
// When adding support for another model, update these constants accordingly.
const IMAGE_SIZE: usize = 224;
const MEAN: [f32; 3] = [0.5, 0.5, 0.5];
const STD: [f32; 3] = [0.5, 0.5, 0.5];

/// SigLIP2 vision encoder ONNX inference example
#[derive(FromArgs)]
#[argh(description = "Run SigLIP2 vision encoder inference via ONNX Runtime")]
struct Args {
    /// path to input image
    #[argh(option)]
    image_path: PathBuf,

    /// path to SigLIP2 ONNX model (download from https://huggingface.co/kornia)
    #[argh(option)]
    onnx_model_path: PathBuf,

    /// path to the ONNX Runtime shared library
    #[argh(option)]
    ort_dylib_path: PathBuf,
}

/// Preprocess an image for SigLIP2 vision encoder.
///
/// Matches the Python `SigLip2ImagePreprocessor` pipeline exactly:
/// 1. Cast u8 -> f32 and rescale to [0, 1]  (Rescale step)
/// 2. Resize to 224x224 using bicubic interpolation  (Resize step)
/// 3. Normalize with mean=[0.5, 0.5, 0.5] and std=[0.5, 0.5, 0.5]  (→ [-1, 1])
/// 4. Permute HWC -> CHW and add batch dim -> [1, 3, 224, 224]
///
/// Reference: https://github.com/kornia/kornia/tree/main/kornia/models/siglip2
fn preprocess(
    img: Image<u8, 3, CpuAllocator>,
) -> Result<Tensor<f32, 4, CpuAllocator>, Box<dyn std::error::Error>> {
    let target_size = ImageSize {
        width: IMAGE_SIZE,
        height: IMAGE_SIZE,
    };

    // step 1: rescale [0, 255] -> [0, 1]
    let mut img_f32 = Image::<f32, 3, _>::from_size_val(img.size(), 0.0, CpuAllocator)?;
    ops::cast_and_scale(&img, &mut img_f32, 1.0 / 255.0)?;

    // step 2: resize to IMAGE_SIZE x IMAGE_SIZE with bicubic interpolation
    let mut img_resized = Image::<f32, 3, _>::from_size_val(target_size, 0.0, CpuAllocator)?;
    resize_native(&img_f32, &mut img_resized, InterpolationMode::Bicubic)?;

    // step 3: normalize using kornia built-in op
    let mut img_norm = Image::<f32, 3, _>::from_size_val(target_size, 0.0, CpuAllocator)?;
    normalize_mean_std(&img_resized, &mut img_norm, &MEAN, &STD)?;

    // step 4: HWC -> CHW, add batch dim -> [1, 3, IMAGE_SIZE, IMAGE_SIZE]
    let chw = img_norm.permute_axes([2, 0, 1]).as_contiguous();
    let nchw = Tensor::from_shape_vec(
        [1, chw.shape[0], chw.shape[1], chw.shape[2]],
        chw.into_vec(),
        CpuAllocator,
    )?;

    Ok(nchw)
}

/// Run SigLIP2 vision encoder via ONNX Runtime and return the embedding shape.
fn run_siglip2(
    input: Tensor<f32, 4, CpuAllocator>,
    model_path: impl AsRef<Path>,
    ort_dylib_path: impl AsRef<Path>,
) -> Result<Vec<usize>, Box<dyn std::error::Error>> {
    // ORT uses this env var for dynamic library loading
    std::env::set_var("ORT_DYLIB_PATH", ort_dylib_path.as_ref());

    let mut session = Session::builder()?
        .with_optimization_level(GraphOptimizationLevel::Level3)?
        .with_intra_threads(4)?
        .commit_from_file(model_path)?;

    let ort_input = OrtTensor::from_array((input.shape, input.into_vec()))?;

    let t0 = Instant::now();
    let outputs = session.run(ort::inputs! {
        "pixel_values" => ort_input,
    })?;
    log::info!("inference time: {:.2} ms", t0.elapsed().as_secs_f32() * 1000.0);

    let embeds = outputs
        .get("image_embeds")
        .ok_or("missing 'image_embeds' in model outputs")?;

    let (shape, _) = embeds.try_extract_tensor::<f32>()?;

    Ok(shape.iter().map(|&x| x as usize).collect())
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    env_logger::init();

    let args: Args = argh::from_env();

    let img = F::read_image_any_rgb8(&args.image_path)?.into_inner();

    let input = preprocess(img)?;

    let embedding_shape = run_siglip2(input, &args.onnx_model_path, &args.ort_dylib_path)?;

    log::info!("embedding shape: {:?}", embedding_shape);

    Ok(())
}
