use argh::FromArgs;
use kornia_tensor::{CpuAllocator, Tensor};
use std::path::PathBuf;
use std::time::Instant;

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

    // set the ort dylib path
    std::env::set_var("ORT_DYLIB_PATH", &args.ort_dylib_path);

    // read the image
    let image: Image<u8, 3> = F::read_image_any_rgb8(&args.image_path)?;

    // read the onnx model

    let model = Session::builder()?
        .with_optimization_level(GraphOptimizationLevel::Level3)?
        .with_intra_threads(4)?
        .commit_from_file(&args.onnx_model_path)?;

    // cast and scale the image to f32
    let mut image_hwc_f32 = Image::from_size_val(image.size(), 0.0f32)?;
    kornia::image::ops::cast_and_scale(&image, &mut image_hwc_f32, 1.0 / 255.0)?;

    // convert to HWC -> CHW
    let image_chw = image_hwc_f32.permute_axes([2, 0, 1]).as_contiguous();

    // TODO: create a Tensor::insert_axis in kornia-rs
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

    // make the ort tensor
    let ort_tensor = ort::value::Tensor::from_array((image_nchw.shape, image_nchw.into_vec()))?;

    println!("ort_tensor: {:?}", ort_tensor.shape());

    // get the inputs names
    let inputs_names = model
        .inputs
        .iter()
        .map(|x| x.name.to_string())
        .collect::<Vec<_>>();

    println!("inputs_name: {:?}", inputs_names);

    // get the outputs names
    let outputs_names = model
        .outputs
        .iter()
        .map(|x| x.name.to_string())
        .collect::<Vec<_>>();

    println!("outputs_name: {:?}", outputs_names);

    let time = Instant::now();

    // run the model
    let outputs = model.run(ort::inputs![
        "input" => ort_tensor,
    ]?)?;

    println!("time ms: {:?}", time.elapsed().as_secs_f32() * 1000.0);

    // get the outputs

    let (out_shape, out_ort) = outputs["output"].try_extract_raw_tensor::<f32>()?;
    println!("out_shape: {:?}", out_shape);

    let out_tensor = Tensor::<f32, 3, CpuAllocator>::from_shape_vec(
        [
            out_shape[0] as usize,
            out_shape[1] as usize,
            out_shape[2] as usize,
        ],
        out_ort.to_vec(),
        CpuAllocator,
    )?;

    println!("out_tensor: {:?}", out_tensor.shape);

    // parse the output tensor
    // we expect the output tensor to be a tensor of shape [1, 300, 6]
    // where each element is a detection [label, score, x, y, w, h]
    let detections = out_tensor
        .as_slice()
        .chunks_exact(6)
        .map(|chunk| Detection {
            label: chunk[0] as u32,
            score: chunk[1],
            x: chunk[2],
            y: chunk[3],
            w: chunk[4],
            h: chunk[5],
        })
        .collect::<Vec<_>>();

    // filter out detections with low confidence
    let detections = detections
        .into_iter()
        .filter(|d| d.score > 0.75)
        .collect::<Vec<_>>();

    // let's log the detections
    let mut boxes_mins = Vec::new();
    let mut boxes_sizes = Vec::new();
    for detection in detections {
        boxes_mins.push((detection.x, detection.y));
        boxes_sizes.push((detection.w, detection.h));
    }

    // create a Rerun recording stream
    let rec = rerun::RecordingStreamBuilder::new("Kornia App").spawn()?;

    rec.log(
        "output",
        &rerun::Image::from_elements(
            image.as_slice(),
            image.size().into(),
            rerun::ColorModel::RGB,
        ),
    )?;

    rec.log(
        "boxes",
        &rerun::Boxes2D::from_mins_and_sizes(boxes_mins, boxes_sizes),
    )?;

    Ok(())
}
