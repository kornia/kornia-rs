use crate::error::IoError;
use image_webp::{ColorType, WebPDecoder, WebPEncoder};
use kornia_image::{
    allocator::{CpuAllocator, ImageAllocator},
    color_spaces::{Gray8, Rgb8, Rgba8},
    Image, ImageLayout, ImageSize, PixelFormat,
};

use std::{
    fs,
    io::{BufReader, Cursor},
    path::Path,
};

// BT.601 luma weights, fixed-point /256: R=0.299, G=0.587, B=0.114.
const GRAY_R: u32 = 77;
const GRAY_G: u32 = 150;
const GRAY_B: u32 = 29;

/// Read a WEBP image as grayscale (Gray8).
///
/// WebP has no native grayscale encoding; the file is decoded as RGB(A) and
/// then converted to luma using BT.601 weights.
///
/// # Arguments
///
/// * `file_path` - The path to the WEBP file.
///
/// # Returns
///
/// A grayscale image (Gray8).
pub fn read_image_webp_gray8(file_path: impl AsRef<Path>) -> Result<Gray8<CpuAllocator>, IoError> {
    let (rgb, size, has_alpha) = read_webp_rgb_or_rgba(file_path)?;
    let gray = rgb_or_rgba_to_gray(&rgb, has_alpha);
    Ok(Gray8::from_size_vec(size, gray, CpuAllocator)?)
}

/// Read a WEBP image as RGB8.
///
/// Returns an error if the file contains an alpha channel.
///
/// # Arguments
///
/// * `file_path` - The path to the WEBP file.
///
/// # Returns
///
/// A RGB8 typed image.
pub fn read_image_webp_rgb8(file_path: impl AsRef<Path>) -> Result<Rgb8<CpuAllocator>, IoError> {
    let (buf, size, has_alpha) = read_webp_rgb_or_rgba(file_path)?;
    if has_alpha {
        return Err(IoError::WebpDecodingError(
            image_webp::DecodingError::InvalidParameter(
                "file has alpha channel; use read_image_webp_rgba8".to_string(),
            ),
        ));
    }
    Ok(Rgb8::from_size_vec(size, buf, CpuAllocator)?)
}

/// Read a WEBP image as RGBA8.
///
/// Returns an error if the file has no alpha channel.
///
/// # Arguments
///
/// * `file_path` - The path to the WEBP file.
///
/// # Returns
///
/// A RGBA8 typed image.
pub fn read_image_webp_rgba8(file_path: impl AsRef<Path>) -> Result<Rgba8<CpuAllocator>, IoError> {
    let (buf, size, has_alpha) = read_webp_rgb_or_rgba(file_path)?;
    if !has_alpha {
        return Err(IoError::WebpDecodingError(
            image_webp::DecodingError::InvalidParameter(
                "file has no alpha channel; use read_image_webp_rgb8".to_string(),
            ),
        ));
    }
    Ok(Rgba8::from_size_vec(size, buf, CpuAllocator)?)
}

/// Decodes a WEBP image as RGB8 from raw bytes.
///
/// Errors if `src` contains an alpha channel or if `dst` dimensions do not match.
///
/// # Arguments
///
/// - `src` - Raw bytes of the webp file
/// - `dst` - A mutable reference to your `Rgb8` image
pub fn decode_image_webp_rgb8<A: ImageAllocator>(
    src: &[u8],
    dst: &mut Image<u8, 3, A>,
) -> Result<(), IoError> {
    decode_webp_impl::<3, _>(src, dst, false)
}

/// Decodes a WEBP image as RGBA8 from raw bytes.
///
/// Errors if `src` has no alpha channel or if `dst` dimensions do not match.
///
/// # Arguments
///
/// - `src` - Raw bytes of the webp file
/// - `dst` - A mutable reference to your `Rgba8` image
pub fn decode_image_webp_rgba8<A: ImageAllocator>(
    src: &[u8],
    dst: &mut Image<u8, 4, A>,
) -> Result<(), IoError> {
    decode_webp_impl::<4, _>(src, dst, true)
}

/// Decodes a WEBP image as Gray8 from raw bytes.
///
/// WebP has no native grayscale encoding; the file is decoded as RGB(A) and
/// converted to luma using BT.601 weights.
///
/// # Arguments
///
/// - `src` - Raw bytes of the webp file
/// - `dst` - A mutable reference to your `Gray8` image
pub fn decode_image_webp_gray8<A: ImageAllocator>(
    src: &[u8],
    dst: &mut Image<u8, 1, A>,
) -> Result<(), IoError> {
    let mut decoder = WebPDecoder::new(Cursor::new(src))?;
    let (width, height) = decoder.dimensions();
    if [width as usize, height as usize] != [dst.width(), dst.height()] {
        return Err(IoError::DecodeMismatchResolution(
            height as usize,
            width as usize,
            dst.height(),
            dst.width(),
        ));
    }

    let buf_size = decoder
        .output_buffer_size()
        .ok_or(IoError::WebpDecodingError(
            image_webp::DecodingError::ImageTooLarge,
        ))?;
    let mut temp_buf = vec![0u8; buf_size];
    decoder.read_image(&mut temp_buf)?;

    let has_alpha = decoder.has_alpha();
    let dst_slice = dst.as_slice_mut();
    let expected_len = (width as usize) * (height as usize);
    if dst_slice.len() != expected_len {
        return Err(IoError::InvalidBufferSize(dst_slice.len(), expected_len));
    }

    let stride = if has_alpha { 4 } else { 3 };
    for (i, chunk) in temp_buf.chunks_exact(stride).enumerate() {
        dst_slice[i] = luma_from_rgb(chunk[0], chunk[1], chunk[2]);
    }
    Ok(())
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
/// Channel count is 3 (RGB) or 4 (RGBA); WebP has no native grayscale encoding.
pub fn decode_image_webp_layout(src: &[u8]) -> Result<ImageLayout, IoError> {
    let decoder = WebPDecoder::new(Cursor::new(src))?;
    let (width, height) = decoder.dimensions();
    let channels: u8 = if decoder.has_alpha() { 4 } else { 3 };
    Ok(ImageLayout::new(
        ImageSize {
            width: width as usize,
            height: height as usize,
        },
        channels,
        PixelFormat::U8,
    ))
}

// Decodes a WEBP image into a pre-allocated Image buffer, validating channel count and size.
fn decode_webp_impl<const C: usize, A: ImageAllocator>(
    src: &[u8],
    dst: &mut Image<u8, C, A>,
    expect_alpha: bool,
) -> Result<(), IoError> {
    let mut decoder = WebPDecoder::new(Cursor::new(src))?;

    let (width, height) = decoder.dimensions();
    if [width as usize, height as usize] != [dst.width(), dst.height()] {
        return Err(IoError::DecodeMismatchResolution(
            height as usize,
            width as usize,
            dst.height(),
            dst.width(),
        ));
    }

    if decoder.has_alpha() != expect_alpha {
        return Err(IoError::WebpDecodingError(
            image_webp::DecodingError::InvalidParameter(format!(
                "channel mismatch: file has_alpha={} but dst expects {} channels",
                decoder.has_alpha(),
                C
            )),
        ));
    }

    let expected_len = (width as usize) * (height as usize) * C;
    let dst_slice = dst.as_slice_mut();
    if dst_slice.len() != expected_len {
        return Err(IoError::InvalidBufferSize(dst_slice.len(), expected_len));
    }

    decoder.read_image(dst_slice)?;
    Ok(())
}

// Reads a WebP file and returns the raw RGB(A) buffer along with size and alpha flag.
fn read_webp_rgb_or_rgba(
    file_path: impl AsRef<Path>,
) -> Result<(Vec<u8>, ImageSize, bool), IoError> {
    let file_path = file_path.as_ref();
    if !file_path.exists() {
        return Err(IoError::FileDoesNotExist(file_path.to_path_buf()));
    }
    match file_path.extension() {
        Some(ext) if ext == "webp" => {}
        _ => return Err(IoError::InvalidFileExtension(file_path.to_path_buf())),
    }

    let file = fs::File::open(file_path)?;
    let reader = BufReader::new(file);
    let mut decoder = WebPDecoder::new(reader)?;
    let buf_size = decoder
        .output_buffer_size()
        .ok_or(IoError::WebpDecodingError(
            image_webp::DecodingError::ImageTooLarge,
        ))?;
    let mut buf = vec![0u8; buf_size];
    decoder.read_image(&mut buf)?;

    let (width, height) = decoder.dimensions();
    let size = ImageSize {
        width: width as usize,
        height: height as usize,
    };
    Ok((buf, size, decoder.has_alpha()))
}

#[inline]
fn luma_from_rgb(r: u8, g: u8, b: u8) -> u8 {
    ((r as u32 * GRAY_R + g as u32 * GRAY_G + b as u32 * GRAY_B) >> 8) as u8
}

fn rgb_or_rgba_to_gray(buf: &[u8], has_alpha: bool) -> Vec<u8> {
    let stride = if has_alpha { 4 } else { 3 };
    buf.chunks_exact(stride)
        .map(|c| luma_from_rgb(c[0], c[1], c[2]))
        .collect()
}

/// Encodes the given RGB8 image to WEBP bytes (VP8L lossless).
///
/// # Arguments
///
/// - `image` - The RGB image to encode
/// - `buffer` - A mutable buffer to write the WEBP bytes into. Existing contents are preserved;
///   the encoded data is appended.
pub fn encode_image_webp_rgb8<A: ImageAllocator>(
    image: &Image<u8, 3, A>,
    buffer: &mut Vec<u8>,
) -> Result<(), IoError> {
    WebPEncoder::new(buffer).encode(
        image.as_slice(),
        image.width() as u32,
        image.height() as u32,
        ColorType::Rgb8,
    )?;
    Ok(())
}

/// Encodes the given RGBA8 image to WEBP bytes (VP8L lossless).
///
/// # Arguments
///
/// - `image` - The RGBA8 image to encode
/// - `buffer` - A mutable buffer to write the WEBP bytes into. Existing contents are preserved;
///   the encoded data is appended.
pub fn encode_image_webp_rgba8<A: ImageAllocator>(
    image: &Image<u8, 4, A>,
    buffer: &mut Vec<u8>,
) -> Result<(), IoError> {
    WebPEncoder::new(buffer).encode(
        image.as_slice(),
        image.width() as u32,
        image.height() as u32,
        ColorType::Rgba8,
    )?;
    Ok(())
}

/// Encodes the given Gray8 image to WEBP bytes (VP8L lossless).
///
/// # Arguments
///
/// - `image` - The Gray8 image to encode
/// - `buffer` - A mutable buffer to write the WEBP bytes into. Existing contents are preserved;
///   the encoded data is appended.
pub fn encode_image_webp_gray8<A: ImageAllocator>(
    image: &Image<u8, 1, A>,
    buffer: &mut Vec<u8>,
) -> Result<(), IoError> {
    WebPEncoder::new(buffer).encode(
        image.as_slice(),
        image.width() as u32,
        image.height() as u32,
        ColorType::L8,
    )?;
    Ok(())
}

/// Writes the given Gray8 image to the given file path as WEBP.
///
/// # Arguments
///
/// - `file_path` - The path to the WEBP image.
/// - `image` - The grayscale image to write
pub fn write_image_webp_gray8<A: ImageAllocator>(
    file_path: impl AsRef<Path>,
    image: &Image<u8, 1, A>,
) -> Result<(), IoError> {
    write_image_webp_impl(file_path, image, ColorType::L8)
}

/// Writes the given RGB8 image to the given file path as WEBP.
///
/// # Arguments
///
/// - `file_path` - The path to the WEBP image.
/// - `image` - The rgb8 image to write
pub fn write_image_webp_rgb8<A: ImageAllocator>(
    file_path: impl AsRef<Path>,
    image: &Image<u8, 3, A>,
) -> Result<(), IoError> {
    write_image_webp_impl(file_path, image, ColorType::Rgb8)
}

/// Writes the given RGBA8 image to the given file path as WEBP.
///
/// # Arguments
///
/// - `file_path` - The path to the WEBP image.
/// - `image` - The rgba8 image to write
pub fn write_image_webp_rgba8<A: ImageAllocator>(
    file_path: impl AsRef<Path>,
    image: &Image<u8, 4, A>,
) -> Result<(), IoError> {
    write_image_webp_impl(file_path, image, ColorType::Rgba8)
}

fn write_image_webp_impl<const N: usize, A: ImageAllocator>(
    file_path: impl AsRef<Path>,
    image: &Image<u8, N, A>,
    color_type: ColorType,
) -> Result<(), IoError> {
    let file = fs::File::create(file_path)?;
    let writer = std::io::BufWriter::new(file);
    WebPEncoder::new(writer).encode(
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
    use std::fs::read;

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
        let file_path = tmp_dir.path().join("fire_write_rgb8.webp");
        let image_data = read_image_webp_rgb8("../../tests/data/fire.webp")?;
        write_image_webp_rgb8(&file_path, &image_data)?;

        let image_data_back = read_image_webp_rgb8(&file_path)?;
        assert!(file_path.exists(), "File does not exist: {file_path:?}");

        assert_eq!(image_data_back.cols(), 320);
        assert_eq!(image_data_back.rows(), 235);
        assert_eq!(image_data_back.num_channels(), 3);
        assert_eq!(image_data.as_slice(), image_data_back.as_slice());

        Ok(())
    }

    #[test]
    fn read_write_webp_rgba8() -> Result<(), IoError> {
        let tmp_dir = tempfile::tempdir()?;
        let file_path = tmp_dir.path().join("synthetic_rgba8.webp");

        let w = 16;
        let h = 8;
        let mut pixels = Vec::with_capacity(w * h * 4);
        for y in 0..h {
            for x in 0..w {
                pixels.extend_from_slice(&[x as u8, y as u8, (x + y) as u8, 0x80]);
            }
        }
        let src = Rgba8::from_size_vec([w, h].into(), pixels, CpuAllocator)?;
        write_image_webp_rgba8(&file_path, &src)?;

        let decoded = read_image_webp_rgba8(&file_path)?;
        assert_eq!(decoded.cols(), w);
        assert_eq!(decoded.rows(), h);
        assert_eq!(decoded.num_channels(), 4);
        assert_eq!(decoded.as_slice(), src.as_slice());

        Ok(())
    }

    #[test]
    fn rejects_non_webp_extension() {
        match read_image_webp_rgb8("../../tests/data/dog.jpeg") {
            Err(IoError::InvalidFileExtension(_)) => {}
            other => panic!("expected InvalidFileExtension, got {:?}", other.err()),
        }
    }

    #[test]
    fn rgb_reader_rejects_rgba_file() -> Result<(), IoError> {
        // Encode a synthetic RGBA webp, then try to read it with the RGB reader.
        let tmp_dir = tempfile::tempdir()?;
        let file_path = tmp_dir.path().join("rgba_only.webp");
        let w = 4;
        let h = 4;
        let pixels = vec![0xAAu8; w * h * 4];
        let src = Rgba8::from_size_vec([w, h].into(), pixels, CpuAllocator)?;
        write_image_webp_rgba8(&file_path, &src)?;

        match read_image_webp_rgb8(&file_path) {
            Err(IoError::WebpDecodingError(_)) => Ok(()),
            Err(other) => panic!("expected WebpDecodingError, got {:?}", other),
            Ok(_) => panic!("expected error, got Ok"),
        }
    }
}
