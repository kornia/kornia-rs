use thiserror::Error;

/// An error type for tensor operations.
#[derive(Error, Debug, PartialEq)]
pub enum KernelError {
    /// Length mismatch for vector operations
    #[error("Length mismatch: expected equal length vectors, got {0} and {1}")]
    LengthMismatch(usize, usize),
}
