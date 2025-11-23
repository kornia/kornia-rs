use crate::error::IoError;
use jpeg_encoder::{ColorType, Encoder};
use kornia_image::{
    allocator::{CpuAllocator, ImageAllocator},
    color_spaces::{Gray8, Rgb8},
    Image, ImageLayout, ImagePixelFormat, ImageSize,
};
use std::{fs, path::Path};

/// Writes the given JPEG _(rgb8)_ data to the given file path.
///
/// # Arguments
///
/// - `file_path` - The path to the JPEG image.
/// - `image` - The RGB8 image to write
/// - `quality` - The quality of the JPEG encoding, range from 0 (lowest) to 100 (highest)
pub fn write_image_jpeg_rgb8<A: ImageAllocator>(
    file_path: impl AsRef<Path>,
    image: &Image<u8, 3, A>,
    quality: u8,
) -> Result<(), IoError> {
    write_image_jpeg_imp(file_path, image, ColorType::Rgb, quality)
}

/// Writes the given JPEG _(grayscale)_ data to the given file path.
///
/// # Arguments
///
/// - `file_path` - The path to the JPEG image.
/// - `image` - The grayscale image to write
/// - `quality` - The quality of the JPEG encoding, range from 0 (lowest) to 100 (highest)
pub fn write_image_jpeg_gray8<A: ImageAllocator>(
    file_path: impl AsRef<Path>,
    image: &Image<u8, 1, A>,
    quality: u8,
) -> Result<(), IoError> {
    write_image_jpeg_imp(file_path, image, ColorType::Luma, quality)
}

fn write_image_jpeg_imp<const N: usize, A: ImageAllocator>(
    file_path: impl AsRef<Path>,
    image: &Image<u8, N, A>,
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

/// Read a JPEG image as RGB8.
///
/// # Arguments
///
/// - `file_path` - The path to the JPEG file.
///
/// # Returns
///
/// An RGB8 typed image.
pub fn read_image_jpeg_rgb8(file_path: impl AsRef<Path>) -> Result<Rgb8<CpuAllocator>, IoError> {
    let img = read_image_jpeg_impl::<3>(file_path)?;
    Ok(Rgb8::from_size_vec(
        img.size(),
        img.into_vec(),
        CpuAllocator,
    )?)
}

/// Reads a JPEG file as grayscale.
///
/// # Arguments
///
/// - `file_path` - The path to the JPEG file.
///
/// # Returns
///
/// A Gray8 typed image.
pub fn read_image_jpeg_mono8(file_path: impl AsRef<Path>) -> Result<Gray8<CpuAllocator>, IoError> {
    let img = read_image_jpeg_impl::<1>(file_path)?;
    Ok(Gray8::from_size_vec(
        img.size(),
        img.into_vec(),
        CpuAllocator,
    )?)
}

/// Decodes a JPEG image with as RGB8 from raw bytes.
///
/// # Arguments
///
/// - `src` - Raw bytes of the jpeg file
/// - `dst` - A mutable reference to your `Rgb8` image
pub fn decode_image_jpeg_rgb8<A: ImageAllocator>(
    src: &[u8],
    dst: &mut Rgb8<A>,
) -> Result<(), IoError> {
    decode_jpeg_impl(src, &mut dst.0)
}

/// Decodes a JPEG image as grayscale (Gray8) from raw bytes.
///
/// # Arguments
///
/// - `src` - Raw bytes of the jpeg file
/// - `dst` - A mutable reference to your `Gray8` image
pub fn decode_image_jpeg_mono8<A: ImageAllocator>(
    src: &[u8],
    dst: &mut Gray8<A>,
) -> Result<(), IoError> {
    decode_jpeg_impl(src, &mut dst.0)
}

fn read_image_jpeg_impl<const N: usize>(
    file_path: impl AsRef<Path>,
) -> Result<Image<u8, N, CpuAllocator>, IoError> {
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

    Ok(Image::new(image_size, img_data, CpuAllocator)?)
}

fn decode_jpeg_impl<const C: usize, A: ImageAllocator>(
    src: &[u8],
    dst: &mut Image<u8, C, A>,
) -> Result<(), IoError> {
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

/// Decodes JPEG image metadata from raw bytes without decoding pixel data.
///
/// # Arguments
///
/// - `src` - Raw bytes of the JPEG file
///
/// # Returns
///
/// An `ImageLayout` containing the image metadata (size, channels, pixel format).
pub fn decode_image_jpeg_info(src: &[u8]) -> Result<ImageLayout, IoError> {
    let mut decoder = zune_jpeg::JpegDecoder::new(src);
    decoder.decode_headers()?;

    let image_info = decoder.info().ok_or_else(|| {
        IoError::JpegDecodingError(zune_jpeg::errors::DecodeErrors::Format(String::from(
            "Failed to find image info from its metadata",
        )))
    })?;

    Ok(ImageLayout::new(
        ImageSize {
            width: image_info.width as usize,
            height: image_info.height as usize,
        },
        image_info.components as u8,
        ImagePixelFormat::U8,
    ))
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
        assert!(file_path.exists(), "File does not exist: {file_path:?}");

        assert_eq!(image_data_back.cols(), 258);
        assert_eq!(image_data_back.rows(), 195);
        assert_eq!(image_data_back.num_channels(), 3);

        Ok(())
    }

    #[test]
    fn decode_jpeg() -> Result<(), IoError> {
        let bytes = read("../../tests/data/dog.jpeg")?;
        let mut image = Rgb8::from_size_val([258, 195].into(), 0, CpuAllocator)?;
        decode_image_jpeg_rgb8(&bytes, &mut image)?;

        assert_eq!(image.cols(), 258);
        assert_eq!(image.rows(), 195);
        assert_eq!(image.num_channels(), 3);

        Ok(())
    }

    #[test]
    fn decode_jpeg_size() -> Result<(), IoError> {
        let bytes = read("../../tests/data/dog.jpeg")?;
        let layout = decode_image_jpeg_info(bytes.as_slice())?;
        assert_eq!(layout.image_size.width, 258);
        assert_eq!(layout.image_size.height, 195);
        assert_eq!(layout.channels, 3);
        Ok(())
    }
}
