use argh::FromArgs;
use kornia_io as io;
use kornia_vlm::paligemma::{Paligemma, PaligemmaConfig};
use kornia_vlm::smol::Smol;
use std::path::PathBuf;

// #[derive(FromArgs)]
// /// Generate a description of an image using Google Paligemma
// struct Args {
//     /// path to an input image
//     #[argh(option, short = 'i')]
//     image_path: PathBuf,

//     /// prompt to ask the model
//     #[argh(option, short = 'p')]
//     text_prompt: String,

//     /// the length of the generated text
//     #[argh(option, default = "100")]
//     sample_length: usize,
// }

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut model = Smol::new();

    model.inference();

    Ok(())
}
