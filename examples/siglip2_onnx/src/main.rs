use argh::FromArgs;
use kornia::image::Image;
use kornia::io::functional as F;
use kornia_image::{allocator::CpuAllocator, ops::cast_and_scale};
use kornia_imgproc::{interpolation::InterpolationMode, resize::resize_fast_rgb};
use kornia_tensor::Tensor;
use ort::session::builder::GraphOptimizationLevel;
use ort::session::Session;
use std::path::PathBuf;
use std::time::Instant;

#[derive(FromArgs)]
/// Zero-shot image classification using SigLIP2 ONNX model
struct Args {
    /// path to input image
    #[argh(option, short = 'i')]
    image_path: PathBuf,

    /// path to SigLIP2 ONNX model
    #[argh(option, short = 'm')]
    model_path: PathBuf,

    /// path to ONNX Runtime dylib
    #[argh(option)]
    ort_dylib_path: PathBuf,

    /// comma-separated class labels to classify
    #[argh(option, short = 'l')]
    labels: String,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Args = argh::from_env();

    std::env::set_var("ORT_DYLIB_PATH", &args.ort_dylib_path);

    println!("🦀 SigLIP2 ONNX Inference Example");
    println!("==================================\n");

    let labels: Vec<&str> = args.labels.split(',').map(|s| s.trim()).collect();
    println!("📋 Labels: {:?}\n", labels);

    println!("📸 Loading image: {:?}", args.image_path);
    let image = F::read_image_any_rgb8(&args.image_path)?;
    println!("   Image size: {:?}\n", image.size());

    println!("⚙️  Preprocessing image...");
    let preprocessed = preprocess_image(&image)?;
    println!("   Preprocessed shape: {:?}\n", preprocessed.shape);

    println!("🔧 Loading ONNX model: {:?}", args.model_path);
    let mut model = Session::builder()?
        .with_optimization_level(GraphOptimizationLevel::Level3)?
        .with_intra_threads(4)?
        .commit_from_file(&args.model_path)?;
    println!("   Model loaded successfully\n");

    let shape_vec: Vec<i64> = preprocessed.shape.iter().map(|&x| x as i64).collect();
    let data_vec: Vec<f32> = preprocessed.as_slice().to_vec();
    let ort_tensor = ort::value::Tensor::from_array((shape_vec.as_slice(), data_vec))?;

    println!("🚀 Running inference...");
    let start = Instant::now();
    let outputs = model.run(ort::inputs!["pixel_values" => ort_tensor])?;
    let inference_time = start.elapsed();
    println!(
        "   Inference time: {:.2}ms\n",
        inference_time.as_secs_f32() * 1000.0
    );

    let (out_shape, _out_data) = outputs["image_embeds"].try_extract_tensor::<f32>()?;
    println!("📊 Output embedding shape: {:?}", out_shape);
    println!("   Embedding dimension: {}\n", out_shape[1]);

    println!("✅ Successfully extracted image embedding!");
    println!("   (Text similarity computation requires text encoder - see README)\n");

    let rec = rerun::RecordingStreamBuilder::new("SigLIP2 ONNX").spawn()?;

    rec.log(
        "input_image",
        &rerun::Image::from_elements(
            image.as_slice(),
            image.size().into(),
            rerun::ColorModel::RGB,
        ),
    )?;

    println!("🎨 Results visualized in Rerun");

    Ok(())
}

fn preprocess_image(
    image: &Image<u8, 3, CpuAllocator>,
) -> Result<Tensor<f32, 4, CpuAllocator>, Box<dyn std::error::Error>> {
    const TARGET_SIZE: usize = 384;

    let target_size = kornia_image::ImageSize {
        width: TARGET_SIZE,
        height: TARGET_SIZE,
    };

    let mut resized = Image::from_size_val(target_size, 0u8, CpuAllocator)?;
    resize_fast_rgb(image, &mut resized, InterpolationMode::Bilinear)?;

    let mut image_f32 = Image::from_size_val(resized.size(), 0.0f32, CpuAllocator)?;
    cast_and_scale(&resized, &mut image_f32, 1.0 / 255.0)?;

    let normalized_data: Vec<f32> = image_f32
        .as_slice()
        .iter()
        .map(|&x| 2.0 * x - 1.0)
        .collect();

    let normalized: Image<f32, 3, CpuAllocator> = Image::new(image_f32.size(), normalized_data, CpuAllocator)?;

    let chw = normalized.permute_axes([2, 0, 1]).as_contiguous();

    let nchw = Tensor::from_shape_vec(
        [1, chw.shape[0], chw.shape[1], chw.shape[2]],
        chw.into_vec(),
        CpuAllocator,
    )?;

    Ok(nchw)
}
