use axum::{
    extract::Query,
    response::{IntoResponse, Json},
};
use indicatif::ParallelProgressIterator;
use rayon::prelude::*;
use serde::Deserialize;
use std::sync::{Arc, Mutex};

#[derive(Debug, Deserialize)]
pub struct MeanStdQuery {
    images_dir: String,
    num_threads: Option<usize>,
}

pub async fn compute_mean_std(query: Query<MeanStdQuery>) -> impl IntoResponse {
    let num_threads = query.num_threads.unwrap_or(1);
    rayon::ThreadPoolBuilder::new()
        .num_threads(num_threads)
        .build()
        .expect("Failed to build thread pool");

    // Walk through the images directory and collect the paths of the images
    let images_paths = walkdir::WalkDir::new(&query.images_dir)
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
        .collect::<Vec<_>>();

    if images_paths.is_empty() {
        println!("No images found in the directory");
        return Json(serde_json::json!({
            "error": "No images found in the directory"
        }));
    }

    println!(
        "🚀 Found {} images. Starting to compute the std and mean !!!",
        images_paths.len()
    );

    // Create a progress bar
    let pb = indicatif::ProgressBar::new(images_paths.len() as u64);
    pb.set_style(indicatif::ProgressStyle::default_bar().template(
        "{spinner:.green} [{elapsed_precise}] [{bar:40.cyan/blue}] {pos:>7}/{len:7} ({eta}) {msg} {per_sec}",
    )
    .expect("Failed to set progress bar style")
    .progress_chars("##>-"));

    // compute the std and mean of the images

    let total_std = Arc::new(Mutex::new(vec![0.0; 3]));
    let total_mean = Arc::new(Mutex::new(vec![0.0; 3]));

    let num_samples = images_paths.len() as f64;

    images_paths
        .into_par_iter()
        .progress_with(pb)
        .for_each(|image_path| {
            // read the image
            let image = kornia_rs::io::functional::read_image_jpeg(&image_path)
                .expect("Failed to read image");

            // compute the std and mean
            let (std, mean) = kornia_rs::core::std_mean(&image);

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

    Json(serde_json::json!({
        "mean": total_mean,
        "std": total_std
    }))
}
