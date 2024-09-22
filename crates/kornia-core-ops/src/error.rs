use thiserror::Error;

/// An error type for tensor operations.
#[derive(Error, Debug, PartialEq)]
pub enum TensorOpsError {
    /// Dimension out of bounds.
    #[error("Dimension out of bounds. The dimension {0} is out of bounds ({1}).")]
    DimOutOfBounds(usize, usize),
}
