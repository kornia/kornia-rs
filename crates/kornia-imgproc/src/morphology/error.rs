/// Errors related to morphological operations.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum MorphologyError {
    /// The provided kernel is empty or improperly sized.
    EmptyKernel,
    /// The kernel rows have inconsistent lengths.
    NonRectangularKernel,
    /// The kernel must have odd dimensions.
    EvenSizedKernel,
    /// The input image has zero width or height.
    EmptyImage,
    /// All elements in the kernel are inactive (false).
    AllKernelElementsInactive,
}