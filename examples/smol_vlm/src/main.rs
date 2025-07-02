use argh::FromArgs;
use kornia_image::Image;
use kornia_tensor::CpuAllocator;
use kornia_vlm::smolvlm::{utils::SmolVlmConfig, SmolVlm};

use kornia_io::jpeg::read_image_jpeg_rgb8;
use std::error::Error;
use std::path::{Path, PathBuf};
use std::{fs, io, io::Write};

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

    /// enable some boolean feature
    #[argh(switch)]
    conversation_style: bool,
}

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
                .unwrap_or("unknown_file.png"),
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
    let args: Args = argh::from_env();

    if args.conversation_style {
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

            model.inference(image, &prompt, args.sample_length, true)?;
        }
    } else {
        // read the image
        let image = read_image_jpeg_rgb8(args.image_path)?;

        // create the paligemma model
        let mut smolvlm = SmolVlm::new(SmolVlmConfig::default())?;

        // generate a caption of the image
        let _caption =
            smolvlm.inference(Some(image), &args.text_prompt, args.sample_length, true)?;
    }

    Ok(())
}
