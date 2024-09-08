#[derive(thiserror::Error, Debug)]
pub enum DnnError {
    #[error("Failed to load model")]
    OrtError(#[from] ort::Error),

    #[error("Image error")]
    ImageError(#[from] kornia_image::ImageError),

    #[error("Tensor error")]
    TensorError(#[from] kornia_core::TensorError),
}
