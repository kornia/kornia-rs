use argh::FromArgs;
use kornia_tensor::CpuAllocator;
use kornia_vlm::smolvlm::{utils::SmolVlmConfig, SmolVlm};

use kornia_io::{jpeg::read_image_jpeg_rgb8, png::read_image_png_rgb8};
use std::path::PathBuf;

#[derive(FromArgs)]
/// Generate a description of an image using SmolVlm
struct Args {
    /// path to an input image
    #[argh(option, short = 'i', default = "PathBuf::new()")]
    image_path: PathBuf,

    /// prompt to ask the model
    #[argh(option, short = 'p', default = "\"\".to_string()")]
    text_prompt: String,

    /// the length of the generated text
    #[argh(option, default = "100")]
    sample_length: usize,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Args = argh::from_env();

    // read the image based on file extension
    let image = match args.image_path.extension().and_then(|ext| ext.to_str()) {
        Some("jpg") | Some("jpeg") => read_image_jpeg_rgb8(&args.image_path).ok(),
        Some("png") => read_image_png_rgb8(&args.image_path).ok(),
        _ => {
            eprintln!("Unsupported image format. Only JPEG and PNG are supported.");
            return Ok(());
        }
    };

    // create the SmolVLM model
    let mut smolvlm = SmolVlm::new(SmolVlmConfig {
        do_sample: false, // set to false for greedy decoding
        seed: 420,
        debug: true,
        ..Default::default()
    })?;

    // generate a caption of the image
    let _caption = smolvlm.inference(&args.text_prompt, image, args.sample_length, CpuAllocator)?;

    Ok(())
}
