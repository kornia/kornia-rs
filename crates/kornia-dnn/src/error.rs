#[derive(thiserror::Error, Debug)]
pub enum DnnError {
    #[error("Please set the ORT_DYLIB_PATH environment variable to the path of the ORT dylib. Error: {0}")]
    OrtDylibError(String),

    #[error("Failed to create ORT session")]
    OrtError(#[from] ort::Error),

    #[error("Image error")]
    ImageError(#[from] kornia_image::ImageError),

    #[error("Tensor error")]
    TensorError(#[from] kornia_core::TensorError),
}
