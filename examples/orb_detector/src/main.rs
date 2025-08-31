use argh::FromArgs;
use std::{path::PathBuf, str::FromStr};

mod static_img;
mod webcam;

/// ORB Detector
#[derive(FromArgs)]
struct Args {
    /// possible values: static, webcam
    #[argh(positional)]
    example_kind: ExampleKind,

    /// path to the image
    #[argh(option, short = 'f')]
    image_path: PathBuf,
}

enum ExampleKind {
    Static,
    Webcam,
}

impl FromStr for ExampleKind {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_lowercase().as_str() {
            "static" => Ok(ExampleKind::Static),
            "webcam" => Ok(ExampleKind::Webcam),
            _ => Err(format!(
                "Invalid example: {s} (possible values: static, webcam)"
            )),
        }
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Args = argh::from_env();

    match args.example_kind {
        ExampleKind::Static => static_img::static_img(&args),
        ExampleKind::Webcam => webcam::webcam(&args),
    }
}
