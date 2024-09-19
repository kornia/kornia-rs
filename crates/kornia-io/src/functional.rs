use std::path::Path;

use kornia_image::{Image, ImageSize};

use crate::error::IoError;

#[cfg(feature = "jpegturbo")]
use super::jpeg::{ImageDecoder, ImageEncoder};

#[cfg(feature = "jpegturbo")]
/// Reads a JPEG image from the given file path.
///
/// The method reads the JPEG image data directly from a file leveraging the libjpeg-turbo library.
///
/// # Arguments
///
/// * `image_path` - The path to the JPEG image.
///
/// # Returns
///
/// An in image containing the JPEG image data.
///
/// # Example
///
/// ```
/// use kornia_image::Image;
/// use kornia_io::functional as F;
///
/// let image: Image<u8, 3> = F::read_image_jpeg("../../tests/data/dog.jpeg").unwrap();
///
/// assert_eq!(image.size().width, 258);
/// assert_eq!(image.size().height, 195);
/// assert_eq!(image.num_channels(), 3);
/// ```
pub fn read_image_jpeg(file_path: impl AsRef<Path>) -> Result<Image<u8, 3>, IoError> {
    let file_path = file_path.as_ref().to_owned();
    // verify the file exists and is a JPEG
    if !file_path.exists() {
        return Err(IoError::FileDoesNotExist(file_path.to_path_buf()));
    }

    if file_path.extension().map_or(true, |ext| {
        ext.to_ascii_lowercase() != "jpg" && ext.to_ascii_lowercase() != "jpeg"
    }) {
        return Err(IoError::InvalidFileExtension(file_path.to_path_buf()));
    }

    // open the file and map it to memory
    let file = std::fs::File::open(file_path)?;
    let mmap = unsafe { memmap2::Mmap::map(&file)? };

    // decode the data directly from memory
    let image: Image<u8, 3> = {
        let mut decoder = ImageDecoder::new()?;
        decoder.decode(&mmap)?
    };

    Ok(image)
}

#[cfg(feature = "jpegturbo")]
/// Writes the given JPEG data to the given file path.
///
/// # Arguments
///
/// * `file_path` - The path to the JPEG image.
/// * `image` - The tensor containing the JPEG image data.
pub fn write_image_jpeg(file_path: impl AsRef<Path>, image: &Image<u8, 3>) -> Result<(), IoError> {
    let file_path = file_path.as_ref().to_owned();

    // compress the image
    let jpeg_data = ImageEncoder::new()?.encode(image)?;

    // write the data directly to a file
    std::fs::write(file_path, jpeg_data)?;

    Ok(())
}

/// Reads an image from the given file path.
///
/// The method tries to read from any image format supported by the image crate.
///
/// # Arguments
///
/// * `file_path` - The path to the image.
///
/// # Returns
///
/// A tensor containing the image data.
///
/// # Example
///
/// ```
/// use kornia_image::Image;
/// use kornia_io::functional as F;
///
/// let image: Image<u8, 3> = F::read_image_any("../../tests/data/dog.jpeg").unwrap();
///
/// assert_eq!(image.size().width, 258);
/// assert_eq!(image.size().height, 195);
/// assert_eq!(image.num_channels(), 3);
/// ```
pub fn read_image_any(file_path: impl AsRef<Path>) -> Result<Image<u8, 3>, IoError> {
    let file_path = file_path.as_ref().to_owned();

    // verify the file exists
    if !file_path.exists() {
        return Err(IoError::FileDoesNotExist(file_path.to_path_buf()));
    }

    // open the file and map it to memory
    let file = std::fs::File::open(file_path)?;
    let mmap = unsafe { memmap2::Mmap::map(&file)? };

    // decode the data directly from memory
    // TODO: update the image crate
    #[allow(deprecated)]
    let img = image::io::Reader::new(std::io::Cursor::new(&mmap))
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
    use crate::functional::read_image_any;

    #[cfg(feature = "jpegturbo")]
    use crate::functional::{read_image_jpeg, write_image_jpeg};

    #[test]
    fn read_any() -> Result<(), IoError> {
        let image = read_image_any("../../tests/data/dog.jpeg")?;
        assert_eq!(image.size().width, 258);
        assert_eq!(image.size().height, 195);
        Ok(())
    }

    #[test]
    #[cfg(feature = "jpegturbo")]
    fn read_jpeg() -> Result<(), IoError> {
        let image = read_image_jpeg("../../tests/data/dog.jpeg")?;
        assert_eq!(image.size().width, 258);
        assert_eq!(image.size().height, 195);
        Ok(())
    }

    #[test]
    #[cfg(feature = "jpegturbo")]
    fn read_write_jpeg() -> Result<(), IoError> {
        let tmp_dir = tempfile::tempdir()?;
        std::fs::create_dir_all(tmp_dir.path())?;

        let file_path = tmp_dir.path().join("dog.jpeg");
        let image_data = read_image_jpeg("../../tests/data/dog.jpeg")?;
        write_image_jpeg(&file_path, &image_data)?;

        let image_data_back = read_image_jpeg(&file_path)?;
        assert!(file_path.exists(), "File does not exist: {:?}", file_path);

        assert_eq!(image_data_back.size().width, 258);
        assert_eq!(image_data_back.size().height, 195);
        assert_eq!(image_data_back.num_channels(), 3);

        Ok(())
    }
}
