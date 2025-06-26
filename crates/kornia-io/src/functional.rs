use crate::{
    error::IoError, jpeg::read_image_jpeg_rgb8, png::read_image_png_rgb8,
    tiff::read_image_tiff_rgb8,
};
use kornia_image::{allocator::CpuAllocator, Image};
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
/// use kornia_image::{Image, allocator::CpuAllocator};
/// use kornia_io::functional as F;
///
/// let image: Image<u8, 3, CpuAllocator> = F::read_image_any_rgb8("../../tests/data/dog.jpeg").unwrap();
///
/// assert_eq!(image.cols(), 258);
/// assert_eq!(image.rows(), 195);
/// assert_eq!(image.num_channels(), 3);
/// ```
pub fn read_image_any_rgb8(
    file_path: impl AsRef<Path>,
) -> Result<Image<u8, 3, CpuAllocator>, IoError> {
    let file_path = file_path.as_ref().to_owned();

    // verify the file exists
    if !file_path.exists() {
        return Err(IoError::FileDoesNotExist(file_path.to_path_buf()));
    }
    if let Some(extension) = file_path.extension() {
        if extension == "jpeg" || extension == "jpg" {
            return read_image_jpeg_rgb8(file_path);
        } else if extension == "png" {
            return read_image_png_rgb8(file_path);
        } else if extension == "tiff" {
            return read_image_tiff_rgb8(file_path);
        }
    }

    Err(IoError::InvalidFileExtension(file_path))
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
