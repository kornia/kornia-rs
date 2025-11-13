use crate::{
    error::IoError, jpeg::read_image_jpeg_rgb8, png::read_image_png_rgb8,
    tiff::read_image_tiff_rgb8,
};
use kornia_image::{allocator::CpuAllocator, color_spaces::Rgb8};
use std::path::Path;

/// Reads a RGB8 image from the given file path.
///
/// The method tries to read from any image format supported by the image crate.
///
/// # Arguments
///
/// * `file_path` - The path to the image.
///
/// # Returns
///
/// A tensor image containing the image data in RGB8 format with shape (H, W, 3).
///
/// # Example
///
/// ```
/// use kornia_io::functional as F;
/// use kornia_image::color_spaces::Rgb8;
///
/// let image: Rgb8<_> = F::read_image_any_rgb8("../../tests/data/dog.jpeg").unwrap();
///
/// assert_eq!(image.cols(), 258);
/// assert_eq!(image.rows(), 195);
/// assert_eq!(image.num_channels(), 3);
/// ```
pub fn read_image_any_rgb8(file_path: impl AsRef<Path>) -> Result<Rgb8<CpuAllocator>, IoError> {
    let file_path = file_path.as_ref().to_owned();

    // verify the file exists
    if !file_path.exists() {
        return Err(IoError::FileDoesNotExist(file_path.to_path_buf()));
    }

    // try to read the image from the file path
    // TODO: handle more image formats
    if let Some(extension) = file_path.extension() {
        match extension.to_string_lossy().to_lowercase().as_ref() {
            "jpeg" | "jpg" => read_image_jpeg_rgb8(file_path),
            "png" => read_image_png_rgb8(file_path),
            "tiff" => read_image_tiff_rgb8(file_path),
            _ => Err(IoError::InvalidFileExtension(file_path)),
        }
    } else {
        Err(IoError::InvalidFileExtension(file_path))
    }
}

#[cfg(test)]
mod tests {
    use crate::error::IoError;
    use crate::functional::read_image_any_rgb8;

    #[test]
    fn read_any() -> Result<(), IoError> {
        let image = read_image_any_rgb8("../../tests/data/dog.jpeg")?;
        assert_eq!(image.cols(), 258);
        assert_eq!(image.rows(), 195);
        Ok(())
    }
}
