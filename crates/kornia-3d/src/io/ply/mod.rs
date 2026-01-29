mod parser;
mod properties;

pub use parser::*;
pub use properties::*;

#[derive(Debug, thiserror::Error)]
pub enum PlyError {
    #[error("Failed to read PLY file")]
    Io(#[from] std::io::Error),

    #[error("Failed to deserialize PLY file")]
    Deserialize(#[from] bincode::error::DecodeError),

    #[error("Unsupported PLY property")]
    UnsupportedProperty,
}
