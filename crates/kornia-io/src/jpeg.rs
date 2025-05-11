use crate::error::IoError;
use jpeg_encoder::{ColorType, Encoder};
use kornia_image::{Image, ImageSize};
use std::{fs, path::Path};

/// Writes the given JPEG _(rgb8)_ data to the given file path.
///
/// # Arguments
///
/// - `file_path` - The path to the JPEG image.
/// - `image` - The tensor containing the JPEG image data
/// - `quality` - The quality of the JPEG encoding, range from 0 (lowest) to 100 (highest)
pub fn write_image_jpeg_rgb8(
    file_path: impl AsRef<Path>,
    image: &Image<u8, 3>,
    quality: u8,
) -> Result<(), IoError> {
    write_image_jpeg_imp(file_path, image, ColorType::Rgb, quality)
}

/// Writes the given JPEG _(grayscale)_ data to the given file path.
///
/// # Arguments
///
/// - `file_path` - The path to the JPEG image.
/// - `image` - The tensor containing the JPEG image data
/// - `quality` - The quality of the JPEG encoding, range from 0 (lowest) to 100 (highest)
pub fn write_image_jpeg_gray8(
    file_path: impl AsRef<Path>,
    image: &Image<u8, 1>,
    quality: u8,
) -> Result<(), IoError> {
    write_image_jpeg_imp(file_path, image, ColorType::Luma, quality)
}

fn write_image_jpeg_imp<const N: usize>(
    file_path: impl AsRef<Path>,
    image: &Image<u8, N>,
    color_type: ColorType,
    quality: u8,
) -> Result<(), IoError> {
    let image_size = image.size();
    let encoder = Encoder::new_file(file_path, quality)?;
    encoder.encode(
        image.as_slice(),
        image_size.width as u16,
        image_size.height as u16,
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
    read_image_jpeg_impl(file_path)
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
pub fn read_image_jpeg_mono8(file_path: impl AsRef<Path>) -> Result<Image<u8, 1>, IoError> {
    read_image_jpeg_impl(file_path)
}

/// Decodes a JPEG image with three channel (rgb8) from Raw Bytes.
///
/// # Arguments
///
/// - `image` - A mutable reference to your `Image`
/// - `bytes` - Raw bytes of the jpeg file
pub fn decode_image_jpeg_rgb8(src: &[u8], dst: &mut Image<u8, 3>) -> Result<(), IoError> {
    decode_jpeg_impl(src, dst)
}

/// Decodes a JPEG image with single channel (mono8) from Raw Bytes.
///
/// # Arguments
///
/// - `image` - A mutable reference to your `Image`
/// - `bytes` - Raw bytes of the jpeg file
pub fn decode_image_jpeg_mono8(src: &[u8], dst: &mut Image<u8, 1>) -> Result<(), IoError> {
    decode_jpeg_impl(src, dst)
}

fn read_image_jpeg_impl<const N: usize>(
    file_path: impl AsRef<Path>,
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

    let jpeg_data = fs::read(file_path)?;
    let mut decoder = zune_jpeg::JpegDecoder::new(jpeg_data);
    decoder.decode_headers()?;

    let image_info = decoder.info().ok_or_else(|| {
        IoError::JpegDecodingError(zune_jpeg::errors::DecodeErrors::Format(String::from(
            "Failed to find image info from its metadata",
        )))
    })?;

    let image_size = ImageSize {
        width: image_info.width as usize,
        height: image_info.height as usize,
    };

    let img_data = decoder.decode()?;

    Ok(Image::new(image_size, img_data)?)
}

fn decode_jpeg_impl<const C: usize>(src: &[u8], dst: &mut Image<u8, C>) -> Result<(), IoError> {
    let mut decoder = zune_jpeg::JpegDecoder::new(src);
    decoder.decode_headers()?;

    let image_info = decoder.info().ok_or_else(|| {
        IoError::JpegDecodingError(zune_jpeg::errors::DecodeErrors::Format(String::from(
            "Failed to find image info from its metadata",
        )))
    })?;

    if [image_info.height as usize, image_info.width as usize] != [dst.height(), dst.width()] {
        return Err(IoError::DecodeMismatchResolution(
            image_info.height as usize,
            image_info.width as usize,
            dst.height(),
            dst.width(),
        ));
    }

    decoder.decode_into(dst.as_slice_mut())?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs::{create_dir_all, read};

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
        write_image_jpeg_rgb8(&file_path, &image_data, 100)?;

        let image_data_back = read_image_jpeg_rgb8(&file_path)?;
        assert!(file_path.exists(), "File does not exist: {:?}", file_path);

        assert_eq!(image_data_back.cols(), 258);
        assert_eq!(image_data_back.rows(), 195);
        assert_eq!(image_data_back.num_channels(), 3);

        Ok(())
    }

    #[test]
    fn decode_jpeg() -> Result<(), IoError> {
        let bytes = read("../../tests/data/dog.jpeg")?;
        let mut image: Image<u8, 3> = Image::from_size_val([258, 195].into(), 0)?;
        decode_image_jpeg_rgb8(&bytes, &mut image)?;

        assert_eq!(image.cols(), 258);
        assert_eq!(image.rows(), 195);
        assert_eq!(image.num_channels(), 3);

        Ok(())
    }
}
