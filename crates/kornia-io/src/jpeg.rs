use crate::error::IoError;
use jpeg_encoder::{ColorType, Encoder};
use kornia_image::{
    allocator::{CpuAllocator, ImageAllocator},
    color_spaces::{Gray8, Rgb8},
    Image, ImageLayout, ImageSize, PixelFormat,
};
use std::{fs, io::Cursor, path::Path};

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

/// Encodes the given RGB8 image to JPEG bytes (in-memory) using a provided buffer.
///
/// This is the zero-allocation version - reuse your buffer across multiple encodes.
///
/// # Arguments
///
/// - `image` - The RGB image to encode
/// - `quality` - The quality of the JPEG encoding, range from 0 (lowest) to 100 (highest)
/// - `buffer` - A mutable buffer to write the JPEG bytes into
///
/// # Note
///
/// The caller is responsible for clearing the buffer if needed. The encoded data will be
/// appended to any existing content in the buffer.
///
/// # Example
///
/// ```rust
/// use kornia_io::jpeg::encode_image_jpeg_rgb8;
/// use kornia_image::{Image, allocator::CpuAllocator};
///
/// let image = Image::<u8, 3, CpuAllocator>::from_size_val([258, 195].into(), 0, CpuAllocator).expect("Failed to create image");
/// let mut buffer = Vec::new();
/// encode_image_jpeg_rgb8(&image, 100, &mut buffer).expect("Failed to encode image");
/// ```
pub fn encode_image_jpeg_rgb8<A: ImageAllocator>(
    image: &Image<u8, 3, A>,
    quality: u8,
    buffer: &mut Vec<u8>,
) -> Result<(), IoError> {
    let encoder = Encoder::new(buffer, quality);
    encoder.encode(
        image.as_slice(),
        image.width() as u16,
        image.height() as u16,
        ColorType::Rgb,
    )?;
    Ok(())
}

/// Encodes the given BGRA8 image to JPEG bytes (in-memory) using a provided buffer.
///
/// This is designed for graphics APIs that use BGRA pixel format (e.g., DirectX, Unreal Engine).
/// The alpha channel is included in the encoding.
///
/// # Arguments
///
/// - `image` - The BGRA image to encode (4 channels: Blue, Green, Red, Alpha)
/// - `quality` - The quality of the JPEG encoding, range from 0 (lowest) to 100 (highest)
/// - `buffer` - A mutable buffer to write the JPEG bytes into
///
/// # Note
///
/// This is the zero-allocation version - reuse your buffer across multiple encodes
/// by calling `buffer.clear()` between encodes. The buffer retains its capacity.
///
/// # Example
///
/// ```no_run
/// use kornia_image::{Image, allocator::CpuAllocator};
/// use kornia_io::jpeg;
///
/// let bgra_data = vec![0u8; 640 * 480 * 4]; // BGRA pixels from graphics API
/// let image = Image::<u8, 4, _>::new([640, 480].into(), bgra_data, CpuAllocator)?;
///
/// let mut buffer = Vec::new();
/// jpeg::encode_image_jpeg_bgra8(&image, 90, &mut buffer)?;
///
/// // Send JPEG bytes over network or save to disk
/// std::fs::write("output.jpg", &buffer)?;
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
pub fn encode_image_jpeg_bgra8<A: ImageAllocator>(
    image: &Image<u8, 4, A>,
    quality: u8,
    buffer: &mut Vec<u8>,
) -> Result<(), IoError> {
    let encoder = Encoder::new(buffer, quality);
    encoder.encode(
        image.as_slice(),
        image.width() as u16,
        image.height() as u16,
        ColorType::Bgra,
    )?;
    Ok(())
}

/// Encodes the given grayscale image to JPEG bytes (in-memory) using a provided buffer.
///
/// This is the zero-allocation version - reuse your buffer across multiple encodes.
///
/// # Arguments
///
/// - `image` - The grayscale image to encode
/// - `quality` - The quality of the JPEG encoding, range from 0 (lowest) to 100 (highest)
/// - `buffer` - A mutable buffer to write the JPEG bytes into
///
/// # Note
///
/// The caller is responsible for clearing the buffer if needed. The encoded data will be
/// appended to any existing content in the buffer.
pub fn encode_image_jpeg_gray8<A: ImageAllocator>(
    image: &Image<u8, 1, A>,
    quality: u8,
    buffer: &mut Vec<u8>,
) -> Result<(), IoError> {
    let encoder = Encoder::new(buffer, quality);
    encoder.encode(
        image.as_slice(),
        image.width() as u16,
        image.height() as u16,
        ColorType::Luma,
    )?;
    Ok(())
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
    dst: &mut Image<u8, 3, A>,
) -> Result<(), IoError> {
    decode_jpeg_impl(src, dst)
}

/// Decodes a JPEG image as grayscale (Gray8) from raw bytes.
///
/// # Arguments
///
/// - `src` - Raw bytes of the jpeg file
/// - `dst` - A mutable reference to your `Gray8` image
pub fn decode_image_jpeg_mono8<A: ImageAllocator>(
    src: &[u8],
    dst: &mut Image<u8, 1, A>,
) -> Result<(), IoError> {
    decode_jpeg_impl(src, dst)
}

fn read_image_jpeg_impl<const N: usize>(
    file_path: impl AsRef<Path>,
) -> Result<Image<u8, N, CpuAllocator>, IoError> {
    use zune_jpeg::zune_core::colorspace::ColorSpace;
    use zune_jpeg::zune_core::options::DecoderOptions;

    let file_path = file_path.as_ref().to_owned();
    if !file_path.exists() {
        return Err(IoError::FileDoesNotExist(file_path.to_path_buf()));
    }

    if file_path
        .extension()
        .is_none_or(|ext| !ext.eq_ignore_ascii_case("jpg") && !ext.eq_ignore_ascii_case("jpeg"))
    {
        return Err(IoError::InvalidFileExtension(file_path.to_path_buf()));
    }

    let jpeg_data = fs::read(file_path)?;

    // First pass: decode headers to get image info
    let mut decoder = zune_jpeg::JpegDecoder::new(Cursor::new(&jpeg_data));
    decoder.decode_headers()?;

    let image_info = decoder.info().ok_or_else(|| {
        IoError::JpegDecodingError(zune_jpeg::errors::DecodeErrors::Format(String::from(
            "Failed to find image info from its metadata",
        )))
    })?;

    // Infer colorspace from actual image components
    let colorspace = match image_info.components {
        1 => ColorSpace::Luma,
        3 => ColorSpace::RGB,
        n => {
            return Err(IoError::JpegDecodingError(
                zune_jpeg::errors::DecodeErrors::Format(format!(
                    "Unsupported JPEG component count: {}. Expected 1 (grayscale) or 3 (RGB)",
                    n
                )),
            ))
        }
    };

    // Validate destination matches image channels
    if image_info.components != N as u8 {
        return Err(IoError::JpegDecodingError(
            zune_jpeg::errors::DecodeErrors::Format(format!(
                "Channel mismatch: JPEG has {} components but requested {}",
                image_info.components, N
            )),
        ));
    }

    let image_size = ImageSize {
        width: image_info.width as usize,
        height: image_info.height as usize,
    };

    // Decode with correct output colorspace
    let options = DecoderOptions::default().jpeg_set_out_colorspace(colorspace);
    let mut decoder = zune_jpeg::JpegDecoder::new_with_options(Cursor::new(&jpeg_data), options);
    let img_data = decoder.decode()?;

    Ok(Image::new(image_size, img_data, CpuAllocator)?)
}

fn decode_jpeg_impl<const C: usize, A: ImageAllocator>(
    src: &[u8],
    dst: &mut Image<u8, C, A>,
) -> Result<(), IoError> {
    use zune_jpeg::zune_core::colorspace::ColorSpace;
    use zune_jpeg::zune_core::options::DecoderOptions;

    // First pass: decode headers to get image info
    let mut decoder = zune_jpeg::JpegDecoder::new(Cursor::new(src));
    decoder.decode_headers()?;

    let image_info = decoder.info().ok_or_else(|| {
        IoError::JpegDecodingError(zune_jpeg::errors::DecodeErrors::Format(String::from(
            "Failed to find image info from its metadata",
        )))
    })?;

    // Infer colorspace from actual image components
    let colorspace = match image_info.components {
        1 => ColorSpace::Luma,
        3 => ColorSpace::RGB,
        n => {
            return Err(IoError::JpegDecodingError(
                zune_jpeg::errors::DecodeErrors::Format(format!(
                    "Unsupported JPEG component count: {}. Expected 1 (grayscale) or 3 (RGB)",
                    n
                )),
            ))
        }
    };

    // Validate destination buffer matches image channels
    if image_info.components != C as u8 {
        return Err(IoError::JpegDecodingError(
            zune_jpeg::errors::DecodeErrors::Format(format!(
                "Channel mismatch: JPEG has {} components but destination expects {}",
                image_info.components, C
            )),
        ));
    }

    if [image_info.height as usize, image_info.width as usize] != [dst.height(), dst.width()] {
        return Err(IoError::DecodeMismatchResolution(
            image_info.height as usize,
            image_info.width as usize,
            dst.height(),
            dst.width(),
        ));
    }

    // Decode with correct output colorspace
    let options = DecoderOptions::default().jpeg_set_out_colorspace(colorspace);
    let mut decoder = zune_jpeg::JpegDecoder::new_with_options(Cursor::new(src), options);
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
pub fn decode_image_jpeg_layout(src: &[u8]) -> Result<ImageLayout, IoError> {
    let mut decoder = zune_jpeg::JpegDecoder::new(Cursor::new(src));
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
        PixelFormat::U8,
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs::{create_dir_all, read};

    #[test]
    fn test_read_jpeg() -> Result<(), IoError> {
        let image = read_image_jpeg_rgb8("../../tests/data/dog.jpeg")?;
        assert_eq!(image.cols(), 258);
        assert_eq!(image.rows(), 195);
        Ok(())
    }

    #[test]
    fn test_read_write_jpeg() -> Result<(), IoError> {
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
    fn test_decode_jpeg() -> Result<(), IoError> {
        let bytes = read("../../tests/data/dog.jpeg")?;
        let mut image = Rgb8::from_size_val([258, 195].into(), 0, CpuAllocator)?;
        decode_image_jpeg_rgb8(&bytes, &mut image)?;

        assert_eq!(image.cols(), 258);
        assert_eq!(image.rows(), 195);
        assert_eq!(image.num_channels(), 3);

        Ok(())
    }

    #[test]
    fn test_decode_jpeg_size() -> Result<(), IoError> {
        let bytes = read("../../tests/data/dog.jpeg")?;
        let layout = decode_image_jpeg_layout(bytes.as_slice())?;
        assert_eq!(layout.image_size.width, 258);
        assert_eq!(layout.image_size.height, 195);
        assert_eq!(layout.channels, 3);
        Ok(())
    }

    #[test]
    fn test_encode_jpeg_rgb8_with_buffer() -> Result<(), IoError> {
        let image = read_image_jpeg_rgb8("../../tests/data/dog.jpeg")?;

        let mut buffer = Vec::new();
        encode_image_jpeg_rgb8(&image, 100, &mut buffer)?;

        // Verify JPEG magic bytes (0xFF 0xD8)
        assert!(buffer.len() > 2, "JPEG output is too small");
        assert_eq!(buffer[0], 0xFF, "Invalid JPEG magic byte 1");
        assert_eq!(buffer[1], 0xD8, "Invalid JPEG magic byte 2");

        // Verify we can decode it back
        let mut decoded: Image<u8, 3, _> =
            Image::from_size_val([258, 195].into(), 0, CpuAllocator)?;
        decode_image_jpeg_rgb8(&buffer, &mut decoded)?;
        assert_eq!(decoded.cols(), 258);
        assert_eq!(decoded.rows(), 195);

        Ok(())
    }

    #[test]
    fn test_encode_jpeg_gray8_with_buffer() -> Result<(), IoError> {
        // Create a synthetic grayscale image for testing
        let image = Image::<u8, 1, _>::from_size_val([258, 195].into(), 128, CpuAllocator)?;

        let mut buffer = Vec::new();
        encode_image_jpeg_gray8(&image, 100, &mut buffer)?;

        // Verify JPEG magic bytes (0xFF 0xD8)
        assert!(buffer.len() > 2, "JPEG output is too small");
        assert_eq!(buffer[0], 0xFF, "Invalid JPEG magic byte 1");
        assert_eq!(buffer[1], 0xD8, "Invalid JPEG magic byte 2");

        // Verify we can decode it back
        let mut decoded: Image<u8, 1, _> =
            Image::from_size_val([258, 195].into(), 0, CpuAllocator)?;
        decode_image_jpeg_mono8(&buffer, &mut decoded)?;
        assert_eq!(decoded.cols(), 258);
        assert_eq!(decoded.rows(), 195);

        Ok(())
    }

    #[test]
    fn test_encode_jpeg_buffer_reuse() -> Result<(), IoError> {
        let image1 = read_image_jpeg_rgb8("../../tests/data/dog.jpeg")?;
        let image2 = Image::<u8, 3, _>::from_size_val([100, 100].into(), 255, CpuAllocator)?;

        // Reuse the same buffer for multiple encodes
        let mut buffer = Vec::new();

        // First encode
        encode_image_jpeg_rgb8(&image1, 100, &mut buffer)?;
        let size1 = buffer.len();
        assert!(size1 > 0, "First encode should produce data");

        // Second encode with different image - buffer should be cleared and reused
        encode_image_jpeg_rgb8(&image2, 100, &mut buffer)?;
        let size2 = buffer.len();
        assert!(size2 > 0, "Second encode should produce data");

        // Verify both magic bytes are correct
        assert_eq!(buffer[0], 0xFF, "Invalid JPEG magic byte 1");
        assert_eq!(buffer[1], 0xD8, "Invalid JPEG magic byte 2");

        // Third encode
        encode_image_jpeg_rgb8(&image1, 90, &mut buffer)?;
        let size3 = buffer.len();
        assert!(size3 > 0, "Third encode should produce data");

        Ok(())
    }
}
