use clap::Parser;
use kornia::core::{CpuAllocator, Tensor};
use kornia::imgproc::interpolation::InterpolationMode;
use std::path::PathBuf;

use kornia::image::{Image, ImageSize};
use kornia::io::functional as F;

#[derive(Parser)]
struct Args {
    #[arg(short, long)]
    image_path: PathBuf,

    #[arg(short, long)]
    onnx_model_path: PathBuf,

    #[arg(long)]
    ort_dylib_path: PathBuf,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = Args::parse();

    // set the ort dylib path
    std::env::set_var("ORT_DYLIB_PATH", &args.ort_dylib_path);

    // read the image
    let image: Image<u8, 3> = F::read_image_any(&args.image_path)?;

    // read the onnx model

    use ort::{GraphOptimizationLevel, Session};

    let model = Session::builder()?
        .with_optimization_level(GraphOptimizationLevel::Level3)?
        .with_intra_threads(4)?
        .commit_from_file(&args.onnx_model_path)?;

    let new_size = [640, 640].into();

    // resize the image
    let mut image_resized = Image::from_size_val(new_size, 0)?;
    kornia::imgproc::resize::resize_fast(&image, &mut image_resized, InterpolationMode::Bilinear)?;

    // cast and scale the image to f32
    let mut image_f32 = Image::from_size_val(image_resized.size(), 0.0)?;
    kornia::image::ops::cast_and_scale(&image_resized, &mut image_f32, 1. / 255.)?;

    // NOTE: this is the old way to do it and seems to work the rest nope
    //let img_t = ndarray::Array3::from_shape_vec((640, 640, 3), image_f32.as_slice().to_vec())?;
    //let img_t = img_t.view().permuted_axes([2, 0, 1]);
    //let img_t = img_t.as_standard_layout().to_owned();
    //let img_t = img_t.insert_axis(ndarray::Axis(0));
    //let binding = img_t.to_owned();
    //let shape = binding.shape();
    //let ort_tensor: ort::Tensor<f32> = ort::Tensor::from_array((shape, img_t.into_raw_vec()))?;

    // create a CHW view
    let image_chw_view = image_f32.permute_axes([2, 0, 1]);

    // make it contiguous
    let mut image_chw_data = Vec::<f32>::with_capacity(3 * new_size.width * new_size.height);

    // TODO: implement Tensor::contiguous in kornia-core
    // Convert the image to contiguous format from HWC to CHW
    for c in 0..image_resized.num_channels() {
        for i in 0..image_resized.rows() {
            for j in 0..image_resized.cols() {
                let value = image_chw_view.get_unchecked([i, j, c]);
                image_chw_data.push(*value);
            }
        }
    }

    // make the rank 4 tensor
    let image_nchw = Tensor::from_shape_vec(
        [1, 3, new_size.height, new_size.width],
        image_chw_data,
        CpuAllocator,
    )?;

    // make the ort tensor
    let ort_tensor: ort::Value<ort::TensorValueType<_>> =
        ort::Tensor::from_array((image_nchw.shape, image_nchw.into_vec()))?;

    //println!("ort_tensor: {:?}", ort_tensor.shape());

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

    // run the model
    let outputs = model.run(ort::inputs![
        "input" => ort_tensor,
    ]?)?;

    // get the outputs

    let (out2_shape, out2_ort) = outputs["2794"].try_extract_raw_tensor::<f32>()?;
    println!("out2_shape: {:?}", out2_shape);

    let (out1_shape, out1_ort) = outputs["output"].try_extract_raw_tensor::<f32>()?;
    println!("out1_shape: {:?}", out1_shape);

    let shape = out1_shape.iter().map(|x| *x as usize).collect::<Vec<_>>();
    let shape: [usize; 3] = shape.try_into().expect("Shape is too big");
    let logits = Tensor::from_shape_slice(shape, out1_ort, CpuAllocator)?;

    let shape = out2_shape.iter().map(|x| *x as usize).collect::<Vec<_>>();
    let shape: [usize; 3] = shape.try_into().expect("Shape is too big");
    let boxes = Tensor::from_shape_slice(shape, out2_ort, CpuAllocator)?;

    let detections = post_process(&logits, &boxes, image_resized.size());

    let detections = detections
        .iter()
        .filter(|d| d.score > 0.8)
        .collect::<Vec<_>>();

    println!("detections: {:?}", detections);

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
            image_resized.as_slice(),
            image_resized.size().into(),
            rerun::ColorModel::RGB,
        ),
    )?;

    rec.log(
        "boxes",
        &rerun::Boxes2D::from_mins_and_sizes(boxes_mins, boxes_sizes),
    )?;

    Ok(())
}

#[derive(Debug)]
pub struct Detection {
    pub x: f32,
    pub y: f32,
    pub w: f32,
    pub h: f32,
    pub score: f32,
    pub label: u32,
}

// https://github.com/lyuwenyu/RT-DETR/blob/main/rtdetrv2_pytorch/src/zoo/rtdetr/rtdetr_postprocessor.py
pub fn post_process(
    logits: &Tensor<f32, 3>,
    boxes: &Tensor<f32, 3>,
    orig_target_size: ImageSize,
) -> Vec<Detection> {
    let mut detections = Vec::new();

    let num_top_queries = 300;
    let num_classes = 80;

    let boxes_slice = boxes.as_slice();

    let mut indexed_scores = logits
        .as_slice()
        .iter()
        .enumerate()
        .map(|(idx, x)| (idx, sigmoid(x)))
        .collect::<Vec<_>>();

    indexed_scores.sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
    let topk_scores = &indexed_scores[..num_top_queries];

    for (index, score) in topk_scores {
        println!("index: {:?}, score: {:?}", index, score);
        let label = index % num_classes;
        let box_idx = index / num_classes;

        let box_i_offset = box_idx * 4;
        let box_i = boxes_slice[box_i_offset..box_i_offset + 4]
            .try_into()
            .unwrap();

        let box_out = box_cxcywh_to_xyxy(&box_i);

        let (x, y, w, h) = (
            box_out[0] * orig_target_size.width as f32,
            box_out[1] * orig_target_size.height as f32,
            box_out[2] * orig_target_size.width as f32,
            box_out[3] * orig_target_size.height as f32,
        );

        let detection = Detection {
            x,
            y,
            w,
            h,
            score: *score,
            label: label as u32,
        };

        detections.push(detection);
    }

    detections
}

fn sigmoid(x: &f32) -> f32 {
    1.0 / (1.0 + f32::exp(-x))
}

fn box_cxcywh_to_xyxy(boxes: &[f32; 4]) -> [f32; 4] {
    let x_c = boxes[0];
    let y_c = boxes[1];
    let w = boxes[2];
    let h = boxes[3];

    let b = [x_c - 0.5 * w, y_c - 0.5 * h, x_c + 0.5 * w, y_c + 0.5 * h];

    b
}
