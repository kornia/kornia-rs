use thiserror::Error;

#[derive(Debug, Error)]
pub enum ResizeError {
    #[error("source or destination has zero width or height")]
    ZeroDimension,
    #[error("buffer size mismatch: expected {expected} bytes, got {got}")]
    BufferSize { expected: usize, got: usize },
    #[error("cubecl runtime error: {0}")]
    Runtime(String),
}
