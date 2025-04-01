use crate::error::IoError;
use image::codecs::jpeg::JpegEncoder;
use image::{ExtendedColorType, ImageEncoder};
use jpeg_decoder::PixelFormat;
use kornia_image::{Image, ImageSize};
use std::fs::File;
use std::path::Path;

/// Writes the given JPEG _(rgb8)_ data to the given file path.
///
/// # Arguments
///
/// - `file_path` - The path to the JPEG image.
/// - `image` - The tensor containing the JPEG image data
pub fn write_image_jpeg_rgb8(
    file_path: impl AsRef<Path>,
    image: &Image<u8, 3>,
) -> Result<(), IoError> {
    write_image_jpeg_internal(file_path, image, ExtendedColorType::Rgb8)
}

/// Writes the given JPEG _(grayscale)_ data to the given file path.
///
/// # Arguments
///
/// - `file_path` - The path to the JPEG image.
/// - `image` - The tensor containing the JPEG image data
pub fn write_image_jpeg_gray8(
    file_path: impl AsRef<Path>,
    image: &Image<u8, 1>,
) -> Result<(), IoError> {
    write_image_jpeg_internal(file_path, image, ExtendedColorType::L8)
}

fn write_image_jpeg_internal<const N: usize>(
    file_path: impl AsRef<Path>,
    image: &Image<u8, N>,
    color_type: ExtendedColorType,
) -> Result<(), IoError> {
    let image_size = image.size();
    let file = File::create(file_path)?;
    let encoder = JpegEncoder::new(file);
    encoder.write_image(
        image.as_slice(),
        image_size.width as u32,
        image_size.height as u32,
        color_type,
    )?;
    Ok(())
}

/// Read a JPEG image with a four channel _(rgb8)_.
///
/// # Arguments
///
/// - `file_path` - The path to the JPEG file.
///
/// # Returns
///
/// A RGB image with four channels _(rgb8)_.
pub fn read_image_jpeg_rgb8(file_path: impl AsRef<Path>) -> Result<Image<u8, 3>, IoError> {
    read_image_jpeg_internal(file_path, PixelFormat::RGB24)
}

/// Reads a JPEG file with a single channel _(mono8)_
///
/// # Arguments
///
/// - `file_path` - The path to the JPEG file.
///
/// # Returns
///
/// A grayscale image with a single channel _(mono8)_.
pub fn read_image_mono8(file_path: impl AsRef<Path>) -> Result<Image<u8, 1>, IoError> {
    read_image_jpeg_internal(file_path, PixelFormat::L8)
}

/// Reads a JPEG file with a single channel _(mono16)_.
///
/// # Arguments
///
/// - `file_path` - The path to the JPEG file.
///
/// # Returns
///
/// A grayscale image with a single channel _(mono16)_.
pub fn read_image_mono16(file_path: impl AsRef<Path>) -> Result<Image<u16, 1>, IoError> {
    let image: Image<u8, 1> = read_image_jpeg_internal(file_path, PixelFormat::L16)?;
    let image_size = image.size();
    let image_buf = image.as_slice();

    let mut buf = Vec::with_capacity(image_buf.len() / 2);
    for img in image_buf {
        buf.push(*img as u16);
    }

    Ok(Image::new(image_size, buf)?)
}

fn read_image_jpeg_internal<const N: usize>(
    file_path: impl AsRef<Path>,
    color_type: PixelFormat,
) -> Result<Image<u8, N>, IoError> {
    let file_path = file_path.as_ref().to_owned();
    if !file_path.exists() {
        return Err(IoError::FileDoesNotExist(file_path.to_path_buf()));
    }

    if file_path.extension().map_or(true, |ext| {
        !ext.eq_ignore_ascii_case("jpg") && !ext.eq_ignore_ascii_case("jpeg")
    }) {
        return Err(IoError::InvalidFileExtension(file_path.to_path_buf()));
    }

    let jpeg_data = File::open(file_path)?;
    let mut decoder = jpeg_decoder::Decoder::new(jpeg_data);
    decoder.read_info().map_err(|e| IoError::JpegError(e))?;

    let image_info = decoder.info().ok_or_else(|| {
        IoError::JpegError(jpeg_decoder::Error::Format(String::from(
            "Failed to found Image Info from it's metadata",
        )))
    })?;

    let pixel_format = image_info.pixel_format;

    let image_size = ImageSize {
        width: image_info.width as usize,
        height: image_info.height as usize,
    };

    let img_data = decoder.decode()?;

    // Convert the image color type, if necessary
    if color_type != pixel_format {
        let mut new_image_data =
            vec![
                0;
                image_info.width as usize * image_info.height as usize * color_type.pixel_bytes()
            ];
        let mut encoder = JpegEncoder::new(&mut new_image_data);
        encoder.encode(
            img_data.as_slice(),
            image_info.width as u32,
            image_info.height as u32,
            pixel_format.into_extended_color_type(),
        )?;

        return Ok(Image::new(image_size, new_image_data)?);
    }

    Ok(Image::new(image_size, img_data)?)
}

trait PixelFormatExt {
    fn into_extended_color_type(self) -> ExtendedColorType;
}

impl PixelFormatExt for PixelFormat {
    fn into_extended_color_type(self) -> ExtendedColorType {
        match self {
            PixelFormat::L16 => ExtendedColorType::L16,
            PixelFormat::L8 => ExtendedColorType::L8,
            PixelFormat::RGB24 => ExtendedColorType::Rgb8,
            PixelFormat::CMYK32 => ExtendedColorType::Cmyk8,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs::create_dir_all;

    #[test]
    fn read_jpeg() -> Result<(), IoError> {
        let image = read_image_jpeg_rgb8("../../tests/data/dog.jpeg")?;
        assert_eq!(image.cols(), 258);
        assert_eq!(image.rows(), 195);
        Ok(())
    }

    #[test]
    fn read_write_jpeg() -> Result<(), IoError> {
        let tmp_dir = tempfile::tempdir()?;
        create_dir_all(tmp_dir.path())?;

        let file_path = tmp_dir.path().join("dog.jpeg");
        let image_data = read_image_jpeg_rgb8("../../tests/data/dog.jpeg")?;
        write_image_jpeg_rgb8(&file_path, &image_data)?;

        let image_data_back = read_image_jpeg_rgb8(&file_path)?;
        assert!(file_path.exists(), "File does not exist: {:?}", file_path);

        assert_eq!(image_data_back.cols(), 258);
        assert_eq!(image_data_back.rows(), 195);
        assert_eq!(image_data_back.num_channels(), 3);

        Ok(())
    }
}
