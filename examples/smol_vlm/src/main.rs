use kornia_image::Image;
use kornia_tensor::CpuAllocator;
use kornia_vlm::smolvlm::{utils::SmolVlmConfig, SmolVlm};

use kornia_io::jpeg::read_image_jpeg_rgb8;
use reqwest;
use std::error::Error;
use std::path::Path;
use std::{fs, io, io::Write};

pub fn load_image_url(
    url: &str,
) -> std::result::Result<Image<u8, 3, CpuAllocator>, Box<dyn Error>> {
    let dir = Path::new(".vscode");

    let file_path = {
        let parsed_url = reqwest::Url::parse(url)?;
        let path = parsed_url.path();
        dir.join(
            Path::new(path)
                .file_name()
                .and_then(|name| name.to_str())
                .unwrap_or("unknown_file.png") // Use PNG as default
                .to_string(),
        )
    };

    if !dir.exists() {
        fs::create_dir(dir).unwrap();
    }

    // Check if the file exists locally
    if file_path.exists() {
        // Use kornia_io to read the JPEG file
        let img = read_image_jpeg_rgb8(&file_path)?;
        println!("Loaded image from local cache.");
        return Ok(img);
    }

    // If the file does not exist, download it and save it
    println!("Downloading image from URL...");

    // Fetch the image as bytes
    let response = reqwest::blocking::get(url)?.bytes()?;
    fs::write(&file_path, &response)?;

    // Use kornia_io to read the JPEG file
    let img = read_image_jpeg_rgb8(&file_path)?;
    println!("Saved image locally as {}", file_path.to_str().unwrap());
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
    let mut model = SmolVlm::new(SmolVlmConfig::default())?;

    // cargo run -p smol_vlm --features cuda
    for _ in 0..10 {
        let img_url = read_input("img> ");
        let image = load_image_url(&img_url)
            .and_then(|v| {
                if model.image_history_count() > 1 {
                    println!("One image max. Cannot add another image. (Restart)");
                    Err(Box::new(io::Error::new(
                        io::ErrorKind::Other,
                        "One image max",
                    )))
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
                |ok| Ok(ok),
            )
            .ok();

        let prompt = read_input("txt> ");

        model.inference(image, &prompt, 1_000, true)?;
    }

    Ok(())
}
