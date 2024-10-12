use kornia_core::TensorError;
use thiserror::Error;

/// An error type for tensor operations.
#[derive(Error, Debug, PartialEq)]
pub enum TensorOpsError {
    /// The dimension to perform the operation over is greater than the number of dimsions of the tensor.
    #[error("Dimension out of bounds. The dimension {0} is out of bounds ({1}).")]
    DimOutOfBounds(usize, usize),

    /// Tensor error
    #[error("Error with the tensor: {0}")]
    TensorError(#[from] TensorError),
}
