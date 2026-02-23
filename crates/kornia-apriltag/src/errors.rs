/// Errors that can occur when working with AprilTag detection.
#[derive(Debug, thiserror::Error)]
pub enum AprilTagError {
    /// Error related to image.
    #[error(transparent)]
    ImageError(#[from] kornia_image::error::ImageError),

    /// The source image dimensions do not match the [TileMinMax](crate::threshold::TileMinMax) dimensions.
    #[error("The source image dimensions do not match the TileMinMax dimensions")]
    ImageTileSizeMismatch,

    /// The minimum image size should be tile_size x tile_size
    #[error("The minimum image size should be tile_size x tile_size")]
    InvalidImageSize,

    /// The image length does not match the UnionFind length.
    #[error("The image length {0} mismatches UnionFind length {1}")]
    InvalidUnionFindSize(usize, usize),

    /// Too many codes for u16 IDs.
    #[error("Too many codes for u16 IDs: {0}")]
    TooManyCodes(usize),

    /// Allowed errors must be less than 4.
    #[error("Allowed errors must be less than 4, got {0}")]
    InvalidAllowedErrors(u8),

    /// Cholesky decomposition failed for grayscale interpolation model.
    /// This typically occurs when the sampled data is insufficient or
    /// the matrix is not positive definite.
    #[error("Cholesky decomposition failed for grayscale interpolation model")]
    GrayModelUnderdetermined,
    /// max_hamming exceeds safe limit for this tag family.
    #[error("max_hamming {max_hamming} exceeds safe limit for this tag family (min_hamming={min_hamming}, max safe value is {max_safe})")]
    MaxHammingTooLarge {
        /// The requested max_hamming value.
        max_hamming: u8,
        /// The minimum Hamming distance between codes in this family.
        min_hamming: u8,
        /// The maximum safe max_hamming value for this family.
        max_safe: u8,
    },
}
