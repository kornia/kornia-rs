use kornia_tensor::{CpuAllocator, Tensor};
use ort::session::builder::GraphOptimizationLevel;
use ort::session::Session;
use ort::value::Tensor as OrtTensor;
use std::path::Path;

pub fn run_siglip2(
    input: Tensor<f32, 4, CpuAllocator>,
    model_path: impl AsRef<Path>,
    ort_dylib_path: impl AsRef<Path>,
) -> Result<Vec<usize>, Box<dyn std::error::Error>> {
    // required by ORT dynamic loader
    std::env::set_var("ORT_DYLIB_PATH", ort_dylib_path.as_ref());

    // session MUST be mutable (ort::Session::run takes &mut self)
    let mut session = Session::builder()?
        .with_optimization_level(GraphOptimizationLevel::Level3)?
        .with_intra_threads(4)?
        .commit_from_file(model_path)?;

    // kornia tensor -> ort tensor
    let ort_input =
        OrtTensor::from_array((input.shape, input.into_vec()))?;

    // run inference
    let outputs = session.run(ort::inputs! {
        "pixel_values" => ort_input,
    })?;

    // extract output
    let embeds = outputs
        .get("image_embeds")
        .ok_or("missing image_embeds output")?;

    let (shape, _) = embeds.try_extract_tensor::<f32>()?;

    Ok(shape.iter().map(|&x| x as usize).collect())
}
