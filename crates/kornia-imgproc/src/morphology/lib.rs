/// Morphological border handling modes and erosion operations.
pub mod morphology;

//
// Re-exports for simplified public API
//

/// Border handling strategies for morphological operations.
pub use morphology::BorderMode;

/// Generates a binary box-shaped kernel of the specified size.
pub use morphology::kernel::generate_box_kernel;

/// Applies a 3x3 erosion using a default binary kernel and ignoring borders.
pub use morphology::erosion::erode;

/// Erosion variants using custom kernels or border handling.
pub mod erode {
    /// Applies erosion using a custom binary kernel, ignoring out-of-bounds pixels.
    pub use crate::morphology::erosion::erode_with_kernel as with_kernel;

    /// Applies erosion using a custom binary kernel and a configurable border mode.
    pub use crate::morphology::erosion::erode_with_kernel_border as with_border;
}
