use kornia_io::jpeg::read_image_jpeg_rgb8;
use kornia_vlm::smolvlm::{utils::SmolVlmConfig, SmolVlm};
use pyo3::prelude::*;

#[pyfunction]
#[pyo3(signature = (text_prompt, sample_length, image_path=None))]
fn generate(
    text_prompt: String,
    sample_length: usize,
    image_path: Option<String>,
) -> PyResult<String> {
    let map_error = |e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("{:?}", e));

    // read the image
    let image = if let Some(image_path) = image_path {
        read_image_jpeg_rgb8(image_path).ok()
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
        .inference(image, &text_prompt, sample_length, true)
        .map_err(map_error)?;

    Ok(caption)
}

/// A Python module implemented in Rust.
#[pymodule]
fn kornia_vlm_pyo3(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(generate, m)?)?;
    Ok(())
}
