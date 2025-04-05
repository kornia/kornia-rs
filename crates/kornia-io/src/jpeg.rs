use crate::error::IoError;
use jpeg_encoder::{ColorType, Encoder};
use kornia_image::{Image, ImageSize};
use kornia_tensor::tensor::get_strides_from_shape;
use std::fs;
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
    write_image_jpeg_imp(file_path, image, ColorType::Rgb)
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
    write_image_jpeg_imp(file_path, image, ColorType::Luma)
}

fn write_image_jpeg_imp<const N: usize>(
    file_path: impl AsRef<Path>,
    image: &Image<u8, N>,
    color_type: ColorType,
) -> Result<(), IoError> {
    let image_size = image.size();
    let encoder = Encoder::new_file(file_path, 100)?;
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
pub fn decode_image_jpeg_rgb8(image: &mut Image<u8, 3>, bytes: &[u8]) -> Result<(), IoError> {
    decode_jpeg_impl(image, bytes)
}

/// Decodes a JPEG image with single channel (mono8) from Raw Bytes.
///
/// # Arguments
///
/// - `image` - A mutable reference to your `Image`
/// - `bytes` - Raw bytes of the jpeg file
pub fn decode_image_jpeg_mono8(image: &mut Image<u8, 1>, bytes: &[u8]) -> Result<(), IoError> {
    decode_jpeg_impl(image, bytes)
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

fn decode_jpeg_impl<const C: usize>(image: &mut Image<u8, C>, bytes: &[u8]) -> Result<(), IoError> {
    let mut decoder = zune_jpeg::JpegDecoder::new(bytes);
    decoder
        .decode_headers()
        .map_err(IoError::JpegDecodingError)?;

    let image_info = decoder.info().ok_or_else(|| {
        IoError::JpegDecodingError(zune_jpeg::errors::DecodeErrors::Format(String::from(
            "Failed to find image info from its metadata",
        )))
    })?;

    let image_size = [image_info.height as usize, image_info.width as usize, C];
    decoder.decode_into(image.as_slice_mut())?;

    // Update the tensor shape and stride
    image.0.shape = image_size;
    image.0.strides = get_strides_from_shape(image_size);
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
        write_image_jpeg_rgb8(&file_path, &image_data)?;

        let image_data_back = read_image_jpeg_rgb8(&file_path)?;
        assert!(file_path.exists(), "File does not exist: {:?}", file_path);

        assert_eq!(image_data_back.cols(), 258);
        assert_eq!(image_data_back.rows(), 195);
        assert_eq!(image_data_back.num_channels(), 3);

        Ok(())
    }

    #[test]
    fn decode_jpeg() -> Result<(), IoError> {
        // This is the size of buffer and must be known before hand
        // for the sake of testing, we are keeping it a constant
        const BUFFER_SIZE: usize = 150930;

        let bytes = read("../../tests/data/dog.jpeg")?;
        let mut image: Image<u8, 3> = Image::new([258, 195].into(), vec![0; BUFFER_SIZE])?;
        decode_image_jpeg_rgb8(&mut image, &bytes)?;

        assert_eq!(image.cols(), 258);
        assert_eq!(image.rows(), 195);
        assert_eq!(image.num_channels(), 3);

        Ok(())
    }
}
