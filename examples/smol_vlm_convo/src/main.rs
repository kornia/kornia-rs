use argh::FromArgs;
use kornia_image::Image;
use kornia_tensor::CpuAllocator;
use kornia_vlm::smolvlm::{utils::SmolVlmConfig, SmolVlm};

use kornia_io::jpeg::read_image_jpeg_rgb8;
use std::error::Error;
use std::path::Path;
use std::{fs, io, io::Write};
use tempfile::TempDir;

#[derive(FromArgs)]
/// Arguments for initializing a traditional conversation-style interface
struct Args {
    /// the length of the generated text
    #[argh(option, default = "100")]
    sample_length: usize,
}

pub fn load_image_url(
    url: &str,
) -> std::result::Result<Image<u8, 3, CpuAllocator>, Box<dyn Error>> {
    // Create a temporary directory that will be cleaned up automatically
    let temp_dir = TempDir::new()?;

    let file_path = {
        let parsed_url = reqwest::Url::parse(url)?;
        let path = parsed_url.path();
        temp_dir.path().join(
            Path::new(path)
                .file_name()
                .and_then(|name| name.to_str())
                .unwrap_or("unknown_file.png"),
        )
    };

    // Check if the file exists locally (unlikely in temp dir, but good practice)
    if file_path.exists() {
        // Use kornia_io to read the JPEG file
        let img = read_image_jpeg_rgb8(&file_path)?;
        println!("Loaded image from temp cache.");
        return Ok(img);
    }

    // Download the image and save it to temp directory
    println!("Downloading image from URL...");

    // Fetch the image as bytes
    let response = reqwest::blocking::get(url)?.bytes()?;
    fs::write(&file_path, &response)?;

    // Use kornia_io to read the JPEG file
    let img = read_image_jpeg_rgb8(&file_path)?;
    println!("Saved image to temp directory: {}", file_path.display());

    // Note: temp_dir will be automatically cleaned up when it goes out of scope
    Ok(img)
}

fn read_input(cli_prompt: &str) -> String {
    let mut input = String::new();
    print!("{}", cli_prompt);
    io::stdout().flush().unwrap();
    io::stdin()
        .read_line(&mut input)
        .expect("Failed to read input");

    input.trim().to_owned()
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Args = argh::from_env();

    let mut model = SmolVlm::new(SmolVlmConfig::default())?;

    for _ in 0..10 {
        let img_url = read_input("img> ");
        let image = load_image_url(&img_url)
            .and_then(|v| {
                if model.image_history_count() > 1 {
                    println!("One image max. Cannot add another image. (Restart)");
                    Err(Box::new(io::Error::other("One image max")))
                } else {
                    Ok(v)
                }
            })
            .map_or_else(
                |err| {
                    println!("Invalid or empty URL (no image)");
                    println!("Error: {:?}", err);

                    Err(err)
                },
                Ok,
            )
            .ok();

        let prompt = read_input("txt> ");

        model.inference(&prompt, image, args.sample_length)?;
    }

    Ok(())
}
