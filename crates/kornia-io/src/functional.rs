use std::{ops::Deref, path::Path};

use kornia_image::{Image, ImageSize, SafeTensorType};

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

/// A generic image type that can be any of the supported image formats.
pub enum GenericImage {
    /// 8-bit grayscale image
    L8(Image<u8, 1>),
    /// 8-bit grayscale image with alpha channel
    La8(Image<u8, 2>),
    /// 8-bit RGB image
    Rgb8(Image<u8, 3>),
    /// 8-bit RGB image with alpha channel
    Rgba8(Image<u8, 4>),
    /// 16-bit grayscale image
    L16(Image<u16, 1>),
    /// 16-bit grayscale image with alpha channel
    La16(Image<u16, 2>),
    /// 16-bit RGB image
    Rgb16(Image<u16, 3>),
    /// 16-bit RGB image with alpha channel
    Rgba16(Image<u16, 4>),
    /// 32-bit float RGB image
    Rgb32F(Image<f32, 3>),
    /// 32-bit float RGB image with alpha channel
    Rgba32F(Image<f32, 4>),
}

// NOTE: another option is to use define types for each of the image formats and then implement
// type Mono8 = Image<u8, 1>;
// type Rgb8 = Image<u8, 3>;

/// Reads an image from the given file path.
///
/// The method tries to read from any image format supported by the image crate.
///
/// # Arguments
///
/// * `file_path` - The path to a valid image file.
///
/// # Returns
///
/// An image containing the image data.
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
pub fn read_image_any<T, const C: usize>(
    file_path: impl AsRef<Path>,
) -> Result<GenericImage, IoError>
where
    T: SafeTensorType,
{
    // resolve the file path correctly
    let file_path = file_path.as_ref().to_owned();

    // verify the file exists
    if !file_path.exists() {
        return Err(IoError::FileDoesNotExist(file_path.to_path_buf()));
    }

    // open the file and map it to memory
    // TODO: explore whether we can use a more efficient memory mapping approach
    let file = std::fs::File::open(file_path)?;
    let mmap = unsafe { memmap2::Mmap::map(&file)? };

    // decode the data directly from memory
    // TODO: update the image crate
    // TODO: explore supporting directly the decoders
    #[allow(deprecated)]
    let img = image::io::Reader::new(std::io::Cursor::new(&mmap))
        .with_guessed_format()?
        .decode()?;

    let size = ImageSize {
        width: img.width() as usize,
        height: img.height() as usize,
    };

    let image = match img.color() {
        image::ColorType::L8 => {
            GenericImage::L8(Image::<u8, 1>::new(size, img.into_luma8().to_vec())?)
        }
        image::ColorType::Rgb8 => {
            GenericImage::Rgb8(Image::<u8, 3>::new(size, img.into_rgb8().to_vec())?)
        }
        _ => return Err(IoError::UnsupportedImageFormat),
    };

    Ok(image)
}

#[cfg(test)]
mod tests {
    use crate::error::IoError;
    use crate::functional::read_image_any;
    use kornia_image::Image;

    #[cfg(feature = "jpegturbo")]
    use crate::functional::{read_image_jpeg, write_image_jpeg};

    #[test]
    fn read_any() -> Result<(), IoError> {
        // let image: Image<u8, 3> = read_image_any("../../tests/data/dog.jpeg")?;
        let image: super::GenericImage = read_image_any("../../tests/data/dog.jpeg")?;
        // NOTE: then how to access the size? we need to reimplment the methods ??
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
