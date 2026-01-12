use argh::FromArgs;
use kornia::io::functional as F;
use std::path::PathBuf;
use std::time::Instant;

mod preprocess;
mod siglip2;

use preprocess::preprocess;
use siglip2::run_siglip2;

/// siglip2 vision onnx example
#[derive(FromArgs)]
#[argh(description = "siglip2 vision onnx example")]
struct Args {
    /// path to input image (224x224 rgb)
    #[argh(option)]
    image_path: PathBuf,

    /// path to siglip2 onnx model
    #[argh(option)]
    onnx_model_path: PathBuf,

    /// path to onnxruntime dynamic library
    #[argh(option)]
    ort_dylib_path: PathBuf,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Args = argh::from_env();

    // ORT uses this env var internally
    std::env::set_var("ORT_DYLIB_PATH", &args.ort_dylib_path);

    // load image
    let img_rgb8 = F::read_image_any_rgb8(&args.image_path)?;
    let img = img_rgb8.as_image();

    // preprocess -> [1,3,224,224]
    let input = preprocess(img.clone());

    let t0 = Instant::now();

    let shape = run_siglip2(
        input,
        &args.onnx_model_path,
        &args.ort_dylib_path,
    )?;

    eprintln!(
        "inference ms: {:.2}",
        t0.elapsed().as_secs_f32() * 1000.0
    );
    eprintln!("embedding shape: {:?}", shape);

    Ok(())
}
