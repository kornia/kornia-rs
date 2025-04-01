use crate::error::IoError;
use image::codecs::jpeg::{JpegDecoder, JpegEncoder};
use image::{ColorType, ExtendedColorType, ImageDecoder, ImageEncoder};
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

/// Writes the given JPEG _(rgba8)_ data to the given file path.
///
/// # Arguments
///
/// - `file_path` - The path to the JPEG image.
/// - `image` - The tensor containing the JPEG image data
pub fn write_image_jpeg_rgba8(
    file_path: impl AsRef<Path>,
    image: &Image<u8, 3>,
) -> Result<(), IoError> {
    write_image_jpeg_internal(file_path, image, ExtendedColorType::Rgba8)
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
    read_image_jpeg_internal(file_path, ColorType::Rgb8)
}

/// Read a JPEG image with a three channel _(rgba8)_.
///
/// # Arguments
///
/// - `file_path` - The path to the JPEG file.
///
/// # Returns
///
/// A RGB image with three channels _(rgba8)_.
pub fn read_image_jpeg_rgba8(file_path: impl AsRef<Path>) -> Result<Image<u8, 3>, IoError> {
    read_image_jpeg_internal(file_path, ColorType::Rgba8)
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
    read_image_jpeg_internal(file_path, ColorType::L8)
}

fn read_image_jpeg_internal<const N: usize>(
    file_path: impl AsRef<Path>,
    color_type: ColorType,
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
    let decoder = JpegDecoder::new(std::io::BufReader::new(jpeg_data))?;
    let (width, height) = decoder.dimensions();
    let decoder_color_type = decoder.color_type();

    let image_size = ImageSize {
        width: width as usize,
        height: height as usize,
    };

    let mut image_data = vec![0; (width * height * color_type.bytes_per_pixel() as u32) as usize];
    decoder.read_image(&mut image_data)?;

    // Convert the image color type, if necessary
    if color_type != decoder_color_type {
        let mut new_image_data =
            vec![0; (width * height * color_type.bytes_per_pixel() as u32) as usize];
        let mut encoder = JpegEncoder::new(&mut new_image_data);
        encoder.encode(
            image_data.as_slice(),
            width,
            height,
            decoder_color_type.into(),
        )?;

        return Ok(Image::new(image_size, new_image_data)?);
    }

    Ok(Image::new(image_size, image_data)?)
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
