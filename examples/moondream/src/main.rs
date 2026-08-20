use argh::FromArgs;
use kornia_vlm::moondream::{Moondream, MoondreamConfig};
use std::path::PathBuf;

#[derive(FromArgs)]
/// Answer a question about an image using Moondream
struct Args {
    /// path to an input image (jpeg or png)
    #[argh(option, short = 'i')]
    image_path: PathBuf,

    /// prompt to ask the model
    #[argh(option, short = 'p')]
    text_prompt: String,

    /// the length of the generated text
    #[argh(option, default = "100")]
    sample_length: usize,

    /// how many times to run inference, for benchmarking
    #[argh(option, default = "1")]
    iterations: usize,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    env_logger::init();
    let args: Args = argh::from_env();

    // read the image based on file extension
    let image = match args
        .image_path
        .extension()
        .and_then(|ext| ext.to_str())
        .map(str::to_lowercase)
        .as_deref()
    {
        Some("jpg") | Some("jpeg") => kornia_io::jpeg::read_image_jpeg_rgb8(&args.image_path)?,
        Some("png") => kornia_io::png::read_image_png_rgb8(&args.image_path)?,
        _ => {
            eprintln!("Unsupported image format. Only JPEG and PNG are supported.");
            return Ok(());
        }
    };

    // create the Moondream model
    let load_start = std::time::Instant::now();
    let mut moondream = Moondream::new(MoondreamConfig::default())?;
    println!("loaded the model in {:?}", load_start.elapsed());

    for i in 0..args.iterations {
        if args.iterations > 1 {
            println!("--- iteration {}/{}", i + 1, args.iterations);
        }
        moondream.inference(&image, &args.text_prompt, args.sample_length, true)?;

        let stats = moondream.stats();
        print!(
            "{} tokens generated, prefill {:?}, decode {:?}",
            stats.generated_tokens, stats.prefill, stats.decode
        );
        match stats.tokens_per_second() {
            Some(tps) => println!(" ({tps:.2} token/s)"),
            None => println!(),
        }
    }

    Ok(())
}
