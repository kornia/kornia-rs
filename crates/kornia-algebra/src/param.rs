//! Local parameterization traits for optimization on manifolds.
use thiserror::Error;

#[derive(Debug, Error)]
pub enum ParamError {
    /// The input global parameter block `x` is too small.
    #[error("global parameter block has wrong size: expected {expected}, got {got}")]
    WrongGlobalSize { expected: usize, got: usize },

    /// The input local update `delta` is too small.
    #[error("local update has wrong size: expected {expected}, got {got}")]
    WrongLocalSize { expected: usize, got: usize },

    /// The output buffer `out` is too small.
    #[error("output buffer has wrong size: expected {expected}, got {got}")]
    WrongOutSize { expected: usize, got: usize },
}

pub trait Param {
    /// Size of the global parameter block (as stored in the variable).
    const GLOBAL_SIZE: usize;

    /// Size of the local update (tangent space dimension).
    const LOCAL_SIZE: usize;

    /// Apply a local update `delta` to a global parameter block `x`.
    fn plus(x: &[f32], delta: &[f32], out: &mut [f32]) -> Result<(), ParamError>;
}
