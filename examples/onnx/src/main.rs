use argh::FromArgs;
use kornia_tensor::{CpuAllocator, Tensor};
use std::path::PathBuf;
use std::time::Instant;
use rerun;

use kornia::image::Image;
use kornia::io::functional as F;

use ort::session::builder::GraphOptimizationLevel;
use ort::session::Session;

/// Represents a detected object in an image.
#[derive(Debug)]
pub struct Detection {
    /// The class label of the detected object.
    pub label: u32,
    /// The confidence score of the detection (typically between 0 and 1).
    pub score: f32,
    /// The x-coordinate of the top-left corner of the bounding box.
    pub x: f32,
    /// The y-coordinate of the top-left corner of the bounding box.
    pub y: f32,
    /// The width of the bounding box.
    pub w: f32,
    /// The height of the bounding box.
    pub h: f32,
}

#[derive(FromArgs)]
/// Perform object detection using ONNX Runtime and log it to Rerun
struct Args {
    /// path to an input image
    #[argh(option, short = 'i')]
    image_path: PathBuf,

    /// path to an ONNX model
    #[argh(option, short = 'm', long = "model-path")]
    onnx_model_path: PathBuf,

    /// path to the ORT dylib
    #[argh(option)]
    ort_dylib_path: PathBuf,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Args = argh::from_env();

    std::env::set_var("ORT_DYLIB_PATH", &args.ort_dylib_path);

    let image = F::read_image_any_rgb8(&args.image_path)?;

    let model = Session::builder()?
        .with_optimization_level(GraphOptimizationLevel::Level3)?
        .with_intra_threads(4)?
        .commit_from_file(&args.onnx_model_path)?;

    let mut image_hwc_f32 =
        Image::from_size_val(image.size(), 0.0f32, CpuAllocator)?;
    kornia::image::ops::cast_and_scale(
        &image,
        &mut image_hwc_f32,
        1.0 / 255.0,
    )?;

    let image_chw = image_hwc_f32
        .permute_axes([2, 0, 1])
        .as_contiguous();

    let image_nchw = Tensor::from_shape_vec(
        [
            1,
            image_chw.shape[0],
            image_chw.shape[1],
            image_chw.shape[2],
        ],
        image_chw.into_vec(),
        CpuAllocator,
    )?;

    let ort_tensor =
        ort::value::Tensor::from_array((image_nchw.shape, image_nchw.into_vec()))?;

    run_detection(model, ort_tensor, image)?;

    Ok(())
}

fn run_detection(
    mut model: Session,
    ort_tensor: ort::value::Tensor<f32>,
    image: Image<u8, 3, CpuAllocator>,
) -> Result<(), Box<dyn std::error::Error>> {

    if model.inputs.len() != 1 || model.outputs.len() != 1 {
        return Err(format!(
            "Expected 1 input and 1 output, got {} inputs and {} outputs",
            model.inputs.len(),
            model.outputs.len()
        ).into());
    }

    let input_name = model.inputs[0].name.as_str();
    let output_name = model.outputs[0].name.as_str();

    let time = Instant::now();

    let outputs = model.run(ort::inputs! {
        input_name => ort_tensor,
    })?;

    println!(
        "inference time ms: {:.2}",
        time.elapsed().as_secs_f32() * 1000.0
    );

    let (out_shape, out_data) =
        outputs[output_name].try_extract_tensor::<f32>()?;

    if out_shape.len() != 3 || out_shape[2] != 6 {
        return Err(format!(
            "Expected output shape [1, N, 6], got {:?}",
            out_shape
        ).into());
    }

    let out_tensor = Tensor::<f32, 3, CpuAllocator>::from_shape_vec(
        [
            out_shape[0] as usize,
            out_shape[1] as usize,
            out_shape[2] as usize,
        ],
        out_data.to_vec(),
        CpuAllocator,
    )?;

    let detections = out_tensor
        .as_slice()
        .chunks_exact(6)
        .filter_map(|c| {
            if c[1] > 0.75 {
                Some(Detection {
                    label: c[0] as u32,
                    score: c[1],
                    x: c[2],
                    y: c[3],
                    w: c[4],
                    h: c[5],
                })
            } else {
                None
            }
        })
        .collect::<Vec<_>>();

    let mut mins = Vec::new();
    let mut sizes = Vec::new();
    for d in detections {
        mins.push((d.x, d.y));
        sizes.push((d.w, d.h));
    }

    let rec = rerun::RecordingStreamBuilder::new("Kornia App").spawn()?;

    rec.log(
        "image",
        &rerun::Image::from_elements(
            image.as_slice(),
            image.size().into(),
            rerun::ColorModel::RGB,
        ),
    )?;

    rec.log(
        "boxes",
        &rerun::Boxes2D::from_mins_and_sizes(mins, sizes),
    )?;

    Ok(())
}