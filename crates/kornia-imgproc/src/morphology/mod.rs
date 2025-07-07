// Morphological image processing operations.
/// Error types used for morphological operations such as erosion.
pub mod error;
pub use error::MorphologyError;

///util function
pub mod utils;

/// Kernel (structuring element) utilities.
pub mod kernel;

/// Erosion operations for grayscale images.
pub mod erosion;

/// Utility function to generate a square binary kernel (structuring element).
pub use kernel::generate_box_kernel;

/// Core erosion functions:
/// - `erode`: 3x3 erosion with default kernel and ignore border handling.
/// - `erode_with_kernel`: Erosion with a custom binary kernel.
/// - `erode_with_kernel_border`: Erosion with custom kernel and configurable border mode.
pub use erosion::{erode, erode_with_kernel, erode_with_kernel_border};