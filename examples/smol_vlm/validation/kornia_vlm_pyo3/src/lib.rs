use kornia_io::jpeg::read_image_jpeg_rgb8;
use kornia_vlm::smolvlm::{utils::SmolVlmConfig, SmolVlm};
use pyo3::prelude::*;

#[pyfunction]
#[pyo3(signature = (text_prompt, sample_length, image_paths=vec![]))]
fn generate(
    text_prompt: String,
    sample_length: usize,
    image_paths: Vec<String>,
) -> PyResult<String> {
    let map_error = |e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("{:?}", e));

    // read the image
    let image = if !image_paths.is_empty() {
        read_image_jpeg_rgb8(&image_paths[0]).ok()
    } else {
        None
    };

    let mut smolvlm = SmolVlm::new(SmolVlmConfig {
        do_sample: false, // set to false for greedy decoding
        seed: 420,
        ..Default::default()
    })
    .map_err(map_error)?;

    // generate a caption of the image
    let caption = smolvlm
        .inference(&text_prompt, image, sample_length)
        .map_err(map_error)?;

    Ok(caption)
}

#[pyfunction]
#[pyo3(signature = (text_prompt, sample_length, image_paths=vec![]))]
fn generate_raw(
    text_prompt: String,
    sample_length: usize,
    image_paths: Vec<String>,
) -> PyResult<String> {
    // NOTE: e is inferred and there's two types of error to handle (SmolVLM & kornia io)
    let map_error = |e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("{:?}", e));
    let map_krn_error = |e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("{:?}", e));

    // read the image
    let images = image_paths
        .into_iter()
        .map(|p| read_image_jpeg_rgb8(&p))
        .collect::<Result<Vec<_>, _>>()
        .map_err(map_krn_error)?;

    let mut smolvlm = SmolVlm::new(SmolVlmConfig {
        do_sample: false, // set to false for greedy decoding
        seed: 420,
        ..Default::default()
    })
    .map_err(map_error)?;

    // generate a caption of the image
    let caption = smolvlm
        .inference_raw(&text_prompt, images, sample_length)
        .map_err(map_error)?;

    Ok(caption)
}

/// A Python module implemented in Rust.
#[pymodule]
fn kornia_vlm_pyo3(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(generate, m)?)?;
    m.add_function(wrap_pyfunction!(generate_raw, m)?)?;
    Ok(())
}
