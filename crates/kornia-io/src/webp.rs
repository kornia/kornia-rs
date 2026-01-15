use crate::error::IoError;
use image_webp::{WebPDecoder, WebPEncoder, ColorType};
use kornia_image::{
    allocator::{CpuAllocator, ImageAllocator},
    color_spaces::{Gray8, Rgb8, Rgba8},
    Image, ImageLayout, ImageSize, PixelFormat,
};

use std::{fs, io::{BufReader, Cursor}, path::Path,};


/// Read a WEBP image as grayscale (Gray8).
///
/// # Arguments
///
/// * `file_path` - The path to the WEBP file.
///
/// # Returns
///
/// A grayscale image (Gray8).
pub fn read_image_webp_gray8(file_path: impl AsRef<Path>)->Result<Gray8<CpuAllocator>, IoError>{
    let img = read_webp_impl::<1>(file_path)?;
    Ok(Gray8::from_size_vec(
        img.size(),
        img.into_vec(),
        CpuAllocator,
    )?)
}

/// Read a WEBP image as RGB8.
///
/// # Arguments
///
/// * `file_path` - The path to the WEBP file.
///
/// # Returns
///
/// A RGB8 typed image.
pub fn read_image_webp_rgb8(file_path: impl AsRef<Path>) -> Result<Rgb8<CpuAllocator>, IoError> {
    let img = read_webp_impl::<3>(file_path)?;
    Ok(Rgb8::from_size_vec(
        img.size(),
        img.into_vec(),
        CpuAllocator,
    )?)
}

/// Read a WEBP image as RGBA8.
///
/// # Arguments
///
/// * `file_path` - The path to the WEBP file.
///
/// # Returns
///
/// A RGBA8 typed image.
pub fn read_image_webp_rgba8(file_path: impl AsRef<Path>) -> Result<Rgba8<CpuAllocator>, IoError> {
    let img = read_webp_impl::<4>(file_path)?;
    Ok(Rgba8::from_size_vec(
        img.size(),
        img.into_vec(),
        CpuAllocator,
    )?)
}

/// Decodes a WEBP image with as RGB8 from raw bytes.
///
/// # Arguments
///
/// - `src` - Raw bytes of the webp file
/// - `dst` - A mutable reference to your `Rgb8` image
pub fn decode_image_webp_rgb8<A: ImageAllocator>(
    src: &[u8],
    dst: &mut Image<u8, 3, A>,
) -> Result<(), IoError> {
    decode_webp_impl(src, dst)
}

/// Decodes a WEBP image with as RGBA8 from raw bytes.
///
/// # Arguments
///
/// - `src` - Raw bytes of the webp file
/// - `dst` - A mutable reference to your `Rgba8` image
pub fn decode_image_webp_rgba8<A: ImageAllocator>(
    src: &[u8],
    dst: &mut Image<u8, 4, A>,
) -> Result<(), IoError> {
    decode_webp_impl(src, dst)
}

/// Decodes a WEBP image with as GRAY8 from raw bytes.
///
/// # Arguments
///
/// - `src` - Raw bytes of the webp file
/// - `dst` - A mutable reference to your `Gray8` image
pub fn decode_image_webp_gray8<A: ImageAllocator>(
    src: &[u8],
    dst: &mut Image<u8, 1, A>,
) -> Result<(), IoError> {
    decode_webp_impl(src, dst)
}

/// Decodes WEBP image metadata from raw bytes without decoding pixel data.
///
/// # Arguments
///
/// - `src` - Raw bytes of the WEBP file
///
/// # Returns
///
/// An `ImageLayout` containing the image metadata (size, channels, pixel format).
pub fn decode_image_webp_layout(src: &[u8]) -> Result<ImageLayout, IoError>{
    let decoder = WebPDecoder::new(Cursor::new(src))
        .map_err(|e| IoError::WebpDecodingError(e))?;

    let (width, height) = decoder.dimensions();

    let pixel_count = (width as usize) * (height as usize);
    let buffer_size = decoder.output_buffer_size()
        .ok_or_else(|| IoError::WebpDecodingError(image_webp::DecodingError::ImageTooLarge))?;

    let channels  = if pixel_count > 0 {
        (buffer_size / pixel_count) as u8
    } else {
        return Err(IoError::WebpDecodingError(image_webp::DecodingError::ImageTooLarge));
    };

    Ok(ImageLayout::new(
        ImageSize {
            width: width as usize,
            height: height as usize,
        },
        channels,
        PixelFormat::U8,
    ))
}

// Wrapped function to decode a WEBP file
fn decode_webp_impl<const C: usize, A: ImageAllocator>(
    src: &[u8],
    dst: &mut Image<u8, C, A>,
) -> Result<(), IoError> {
    let mut decoder = WebPDecoder::new(Cursor::new(src))
        .map_err(|e| IoError::WebpDecodingError(e))?;

    // Validate Dimensions
    let (width, height) = decoder.dimensions();
    if [width as usize, height as usize] != [dst.width(), dst.height()] {
        return Err(IoError::WebpDecodingError(
            image_webp::DecodingError::InconsistentImageSizes
        ));
    }

    // Grayscale Conversion
    if C == 1 {        
        let buff_size = decoder.output_buffer_size()
            .ok_or(IoError::WebpDecodingError(image_webp::DecodingError::ImageTooLarge))?;
            
        let mut temp_buf = vec![0u8; buff_size];
        decoder.read_image(&mut temp_buf)
            .map_err(|e| IoError::WebpDecodingError(e))?;

        let has_alpha = decoder.has_alpha();
        let dst_slice = dst.as_slice_mut();

        let expected_len = (width as usize) * (height as usize);
        if dst_slice.len() != expected_len {
            return Err(IoError::WebpDecodingError(
                 image_webp::DecodingError::InvalidParameter("Destination buffer size mismatch for grayscale conversion".to_string())
             ));
        }

        if has_alpha {
            // RGBA to GRAY
            for (i, chunk) in temp_buf.chunks_exact(4).enumerate() {
                // p[0]=R, p[1]=G, p[2]=B, p[3]=A
                dst_slice[i] = ((chunk[0] as u32 * 54 + chunk[1] as u32 * 183 + chunk[2] as u32 * 19) >> 8) as u8;
            }
        } else {
            // RGB to Gray
            for (i, chunk) in temp_buf.chunks_exact(3).enumerate() {
                dst_slice[i] = ((chunk[0] as u32 * 54 + chunk[1] as u32 * 183 + chunk[2] as u32 * 19) >> 8) as u8;
            }
        }

    } else {
        let required_size = decoder.output_buffer_size().unwrap_or(0);
        if dst.as_slice_mut().len() != required_size {
            return Err(IoError::WebpDecodingError(
                image_webp::DecodingError::InvalidParameter(format!(
                    "Channel mismatch: WebP needs buffer of size {}, but dst is {}", 
                    required_size, dst.as_slice_mut().len()
                ))
             ));
        }

        decoder.read_image(dst.as_slice_mut())
            .map_err(|e| IoError::WebpDecodingError(e))?;
    }

    Ok(())
}

// Utility function to read a WEBP file
fn read_webp_impl<const N : usize>(file_path : impl AsRef<Path>)
-> Result<Image<u8, N, CpuAllocator>, IoError>{
    // Verifying the File Exists
    let file_path = file_path.as_ref();
    if !file_path.exists(){
        return  Err(IoError::FileDoesNotExist(file_path.to_path_buf()));
    }
    // Verifying the file extension
    if let Some(extension) = file_path.extension(){
        if extension != "webp"{
            return Err(IoError::InvalidFileExtension(file_path.to_path_buf()));
        }
    }
    else{
        return Err(IoError::InvalidFileExtension(file_path.to_path_buf()));
    }
    
    let file = fs::File::open(file_path)?;
    let file_reader = BufReader::new(file);
    let mut decoder = WebPDecoder::new(file_reader)
        .map_err(|e| IoError::WebpDecodingError(e))?;
    let buff_size = decoder.output_buffer_size()
        .ok_or_else(|| IoError::WebpDecodingError(image_webp::DecodingError::ImageTooLarge))?;
    let mut image_data = vec![0u8; buff_size];
    decoder.read_image(&mut image_data)?;

    // Gray8 conversion from rgb or rgba on N = 1
    let final_data = if N == 1 {
        if decoder.has_alpha() {
            // RGBA to Gray8
            image_data.chunks_exact(4)
                .map(|p| {
                    // We ignore p[3] (Alpha) and focus on RGB
                    ((p[0] as u32 * 54 + p[1] as u32 * 183 + p[2] as u32 * 19) >> 8) as u8
                })
                .collect()
        } else {
            // RGB to Gray8
            image_data.chunks_exact(3)
                .map(|p| {
                    ((p[0] as u32 * 54 + p[1] as u32 * 183 + p[2] as u32 * 19) >> 8) as u8
                })
                .collect()
        }
    } else {
        image_data
    };

    let (width, height) = decoder.dimensions();
    let image_size = ImageSize {
        width: width as usize,
        height: height as usize,
    };


    
    Ok(Image::new(image_size, final_data, CpuAllocator)?)
    
}

/// Encodes the given RGB8 image to WEBP bytes (in-memory) using a provided buffer.
///
/// The encoder uses VP8L (Lossless Encoding)
/// 
/// # Arguments
///
/// - `image` - The RGB image to encode
/// - `buffer` - A mutable buffer to write the WEBP bytes into
///
/// # Note
///
/// The caller is responsible for clearing the buffer if needed. The encoded data will be
/// appended to any existing content in the buffer.
pub fn encode_image_webp_rgb8<A: ImageAllocator>(
    image: &Image<u8, 3, A>,
    buffer: &mut Vec<u8>,
) -> Result<(), IoError>{
    let encoder = WebPEncoder::new(buffer);
    encoder.encode(
        image.as_slice(),
        image.width() as u32,
        image.height() as u32,
        ColorType::Rgb8
    )?;
    Ok(())
}


/// Encodes the given RGBA8 image to WEBP bytes (in-memory) using a provided buffer.
///
/// The encoder uses VP8L (Lossless Encoding)
/// 
/// # Arguments
///
/// - `image` - The RGBA8 image to encode
/// - `buffer` - A mutable buffer to write the WEBP bytes into
///
/// # Note
///
/// The caller is responsible for clearing the buffer if needed. The encoded data will be
/// appended to any existing content in the buffer.
pub fn encode_image_webp_rgba8<A: ImageAllocator>(
    image: &Image<u8, 4, A>,
    buffer: &mut Vec<u8>,
) -> Result<(), IoError>{
    let encoder = WebPEncoder::new(buffer);
    encoder.encode(
        image.as_slice(),
        image.width() as u32,
        image.height() as u32,
        ColorType::Rgba8
    )?;
    Ok(())
}

/// Encodes the given GRAY8 image to WEBP bytes (in-memory) using a provided buffer.
///
/// The encoder uses VP8L (Lossless Encoding)
/// 
/// # Arguments
///
/// - `image` - The GRAY image to encode
/// - `buffer` - A mutable buffer to write the WEBP bytes into
///
/// # Note
///
/// The caller is responsible for clearing the buffer if needed. The encoded data will be
/// appended to any existing content in the buffer.
pub fn encode_image_webp_gray8<A: ImageAllocator>(
    image: &Image<u8, 1, A>,
    buffer: &mut Vec<u8>,
) -> Result<(), IoError>{
    let encoder = WebPEncoder::new(buffer);
    encoder.encode(
        image.as_slice(),
        image.width() as u32,
        image.height() as u32,
        ColorType::L8
    )?;
    Ok(())
}

/// Writes the given WEBP _(grayscale)_ data to the given file path.
///
/// # Arguments
///
/// - `file_path` - The path to the WEBP image.
/// - `image` - The grayscale image to write
pub fn write_image_webp_gray8<A: ImageAllocator>(
    file_path: impl AsRef<Path>,
    image: &Image<u8, 1, A>,
) -> Result<(), IoError> {
    write_image_webp_imp(file_path, image, ColorType::L8)
}

/// Writes the given WEBP _(rgb8)_ data to the given file path.
///
/// # Arguments
///
/// - `file_path` - The path to the WEBP image.
/// - `image` - The rgb8 image to write
pub fn write_image_webp_rgb8<A: ImageAllocator>(
    file_path: impl AsRef<Path>,
    image: &Image<u8, 3, A>,
) -> Result<(), IoError> {
    write_image_webp_imp(file_path, image, ColorType::Rgb8)
}

/// Writes the given WEBP _(rgba8)_ data to the given file path.
///
/// # Arguments
///
/// - `file_path` - The path to the WEBP image.
/// - `image` - The rgba8 image to write
pub fn write_image_webp_rgba8<A: ImageAllocator>(
    file_path: impl AsRef<Path>,
    image: &Image<u8, 4, A>,
) -> Result<(), IoError> {
    write_image_webp_imp(file_path, image, ColorType::Rgba8)
}

fn write_image_webp_imp<const N: usize, A: ImageAllocator>(
    file_path: impl AsRef<Path>,
    image: &Image<u8, N, A>,
    color_type: ColorType,
) -> Result<(), IoError> {

    let file = fs::File::create(file_path).map_err(image_webp::EncodingError::IoError)?;
    let writer = std::io::BufWriter::new(file);
    let encoder = WebPEncoder::new(writer);
    encoder.encode(
        image.as_slice(),
        image.width() as u32,
        image.height() as u32,
        color_type,
    )?;
    Ok(())
}


#[cfg(test)]
mod tests {
    use super::*;
    use std::fs::{read};

    #[test]
    fn test_read_webp_rgb8() -> Result<(), IoError> {
        let image = read_image_webp_rgb8("../../tests/data/fire.webp")?;
        assert_eq!(image.cols(), 320);
        assert_eq!(image.rows(), 235);
        Ok(())
    }

    #[test]
    fn test_read_webp_gray8() -> Result<(), IoError> {
        let image = read_image_webp_gray8("../../tests/data/fire.webp")?;
        assert_eq!(image.cols(), 320);
        assert_eq!(image.rows(), 235);
        Ok(())
    }

    #[test]
    fn test_decode_webp() -> Result<(), IoError> {
        let bytes = read("../../tests/data/fire.webp")?;
        let mut image = Rgb8::from_size_val([320, 235].into(), 0, CpuAllocator)?;
        decode_image_webp_rgb8(&bytes, &mut image)?;

        assert_eq!(image.cols(), 320);
        assert_eq!(image.rows(), 235);
        assert_eq!(image.num_channels(), 3);

        Ok(())
    }

    #[test]
    fn test_decode_webp_layout_size() -> Result<(), IoError> {
        let bytes = read("../../tests/data/fire.webp")?;
        let layout = decode_image_webp_layout(bytes.as_slice())?;
        assert_eq!(layout.image_size.width, 320);
        assert_eq!(layout.image_size.height, 235);
        assert_eq!(layout.channels, 3);
        Ok(())
    }

    #[test]
    fn read_write_webp_rgb8() -> Result<(), IoError> {
        let tmp_dir = tempfile::tempdir()?;
        fs::create_dir_all(tmp_dir.path())?;

        let file_path = tmp_dir.path().join("fire_write_rgb8.webp");
        let image_data = read_image_webp_rgb8("../../tests/data/fire.webp")?;
        write_image_webp_rgb8(&file_path, &image_data)?;

        let image_data_back = read_image_webp_rgb8(&file_path)?;
        assert!(file_path.exists(), "File does not exist: {file_path:?}");

        assert_eq!(image_data_back.cols(), 320);
        assert_eq!(image_data_back.rows(), 235);
        assert_eq!(image_data_back.num_channels(), 3);

        Ok(())
    }

}