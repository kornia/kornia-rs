use std::{
    path::PathBuf,
    sync::{Arc, Mutex},
};

use argh::FromArgs;
use indicatif::{ParallelProgressIterator, ProgressStyle};
use rayon::iter::{IntoParallelIterator, ParallelIterator};

use kornia::imgproc;
use kornia::io::functional as F;

// #[derive(Parser, Debug)]
#[derive(FromArgs, Debug)]
/// Compute the std and mean of a set of images.
struct Args {
    /// path to the directory containing the images
    #[argh(option, short = 'i')]
    images_dir: PathBuf,

    /// number of threads to use
    #[argh(option, short = 'n', default = "8")]
    num_threads: usize,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Args = argh::from_env();

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
        "{spinner:.green} [{elapsed_precise}] [{bar:40.cyan/blue}] {pos:>7}/{len:7} ({eta}) {msg} {per_sec}",
    )?.progress_chars("##>-"));

    // compute the std and mean of the images

    let total_std = Arc::new(Mutex::new(vec![0.0; 3]));
    let total_mean = Arc::new(Mutex::new(vec![0.0; 3]));

    let num_samples = images_paths.len() as f64;

    images_paths
        .into_par_iter()
        .progress_with(pb)
        .for_each(|image_path| {
            // read the image
            let image = F::read_image_any_rgb8(image_path).expect("Failed to read image");

            // compute the std and mean
            let (std, mean) = imgproc::core::std_mean(&image);

            // update the total std and mean

            total_std
                .lock()
                .expect("Failed to lock total std")
                .iter_mut()
                .zip(std.iter())
                .for_each(|(t, s)| *t += s);

            total_mean
                .lock()
                .expect("Failed to lock total mean")
                .iter_mut()
                .zip(mean.iter())
                .for_each(|(t, m)| *t += m);
        });

    // average the measurements
    let total_std = total_std
        .lock()
        .expect("Failed to lock total std")
        .iter()
        .map(|&s| s / num_samples)
        .collect::<Vec<_>>();
    let total_mean = total_mean
        .lock()
        .expect("Failed to lock total mean")
        .iter()
        .map(|&m| m / num_samples)
        .collect::<Vec<_>>();

    println!("🔥Total std: {:?}", total_std);
    println!("🔥Total mean: {:?}", total_mean);

    Ok(())
}
