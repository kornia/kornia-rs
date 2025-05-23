use crate::error::IoError;
use kornia_image::{Image, ImageSize};
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
/// use kornia_image::Image;
/// use kornia_io::functional as F;
///
/// let image: Image<u8, 3> = F::read_image_any_rgb8("../../tests/data/dog.jpeg").unwrap();
///
/// assert_eq!(image.cols(), 258);
/// assert_eq!(image.rows(), 195);
/// assert_eq!(image.num_channels(), 3);
/// ```
pub fn read_image_any_rgb8(file_path: impl AsRef<Path>) -> Result<Image<u8, 3>, IoError> {
    let file_path = file_path.as_ref().to_owned();

    // verify the file exists
    if !file_path.exists() {
        return Err(IoError::FileDoesNotExist(file_path.to_path_buf()));
    }

    // open the file and map it to memory
    let jpeg_data = std::fs::read(file_path)?;

    // decode the data directly from memory
    let img = image::ImageReader::new(std::io::Cursor::new(&jpeg_data))
        .with_guessed_format()?
        .decode()?;

    // TODO: handle more image formats
    // return the image data
    let image = Image::new(
        ImageSize {
            width: img.width() as usize,
            height: img.height() as usize,
        },
        img.to_rgb8().to_vec(),
    )?;

    Ok(image)
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
