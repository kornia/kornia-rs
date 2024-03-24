use std::path::PathBuf;

use anyhow::Result;
use clap::Parser;
use indicatif::{ParallelProgressIterator, ProgressStyle};
use rayon::iter::{IntoParallelRefIterator, ParallelIterator};

use kornia_rs::io::functional as F;

#[derive(Parser, Debug)]
struct Args {
    #[arg(short, long)]
    images_dir: PathBuf,

    #[arg(short, long, default_value = "8")]
    num_threads: usize,
}

fn main() -> Result<()> {
    let args = Args::parse();

    rayon::ThreadPoolBuilder::new()
        .num_threads(args.num_threads)
        .build_global()
        .expect("Failed to build thread pool");

    // Walk through the images directory and collect the paths of the images
    let images_paths: Vec<PathBuf> = walkdir::WalkDir::new(&args.images_dir)
        .into_iter()
        .filter_map(|entry| entry.ok())
        .filter(|entry| {
            entry.file_type().is_file()
                && entry
                    .path()
                    .extension()
                    .map(|ext| ext == "jpeg")
                    .unwrap_or(false)
        })
        .map(|entry| entry.path().to_path_buf())
        .collect();

    if images_paths.is_empty() {
        println!("No images found in the directory");
        return Ok(());
    }

    println!(
        "🚀 Found {} images. Starting to compute the std and mean !!!",
        images_paths.len()
    );

    // Create a progress bar
    let pb = indicatif::ProgressBar::new(images_paths.len() as u64);
    pb.set_style(ProgressStyle::default_bar().template(
        "{spinner:.green} [{elapsed_precise}] [{bar:40.cyan/blue}] {pos:>7}/{len:7} ({eta}) {msg} {per_sec} img/sec",
    )?.progress_chars("##>-"));

    // compute the std and mean of the images
    let (total_std, total_mean) = images_paths
        .par_iter()
        .progress_with(pb)
        .map(|image_path| {
            // read the image
            let image = F::read_image_jpeg(image_path).expect("Failed to read image");
            // compute the std and mean
            kornia_rs::core::std_mean(&image)
        })
        .reduce(
            || (vec![0.0; 3], vec![0.0; 3]),
            |(mut total_std, mut total_mean), (std, mean)| {
                total_std
                    .iter_mut()
                    .zip(std.iter())
                    .for_each(|(t, s)| *t += s);
                total_mean
                    .iter_mut()
                    .zip(mean.iter())
                    .for_each(|(t, m)| *t += m);

                (total_std, total_mean)
            },
        );

    // average the measurements
    let n = images_paths.len() as f64;
    let total_std = total_std.iter().map(|&s| s / n).collect::<Vec<_>>();
    let total_mean = total_mean.iter().map(|&m| m / n).collect::<Vec<_>>();

    println!("🔥Total std: {:?}", total_std);
    println!("🔥Total mean: {:?}", total_mean);

    Ok(())
}
