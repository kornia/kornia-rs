mod parser;
mod properties;

pub use parser::*;
pub use properties::*;

/// Error types for the PLY module.
#[derive(Debug, thiserror::Error)]
pub enum PlyError {
    /// Failed to read PLY file
    #[error("Failed to read PLY file")]
    Io(#[from] std::io::Error),

    /// Failed to deserialize PLY file
    #[error("Failed to deserialize PLY file")]
    Deserialize(#[from] bincode::Error),

    /// Unsupported PLY property
    #[error("Unsupported PLY property")]
    UnsupportedProperty,
}
