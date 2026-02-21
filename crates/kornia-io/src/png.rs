use crate::{
    conv_utils::{convert_buf_u16_u8, convert_buf_u8_u16, convert_buf_u8_u16_into_slice},
    error::IoError,
};
use kornia_image::{
    allocator::{CpuAllocator, ImageAllocator},
    color_spaces::{Gray16, Gray8, Rgb16, Rgb8, Rgba16, Rgba8},
    Image, ImageLayout, ImageSize, PixelFormat,
};
use png::{BitDepth, ColorType, Decoder, Encoder};
use std::{
    fs,
    fs::File,
    io::{BufReader, Cursor},
    path::Path,
};

/// Read a PNG image as grayscale (Gray8).
///
/// # Arguments
///
/// * `file_path` - The path to the PNG file.
///
/// # Returns
///
/// A grayscale image (Gray8).
pub fn read_image_png_mono8(file_path: impl AsRef<Path>) -> Result<Gray8<CpuAllocator>, IoError> {
    let (buf, size) = read_png_impl(file_path)?;
    Ok(Gray8::from_size_vec(size.into(), buf, CpuAllocator)?)
}

/// Read a PNG image as RGB8.
///
/// # Arguments
///
/// * `file_path` - The path to the PNG file.
///
/// # Returns
///
/// An RGB8 typed image.
pub fn read_image_png_rgb8(file_path: impl AsRef<Path>) -> Result<Rgb8<CpuAllocator>, IoError> {
    let (buf, size) = read_png_impl(file_path)?;
    Ok(Rgb8::from_size_vec(size.into(), buf, CpuAllocator)?)
}

/// Read a PNG image as RGBA8.
///
/// # Arguments
///
/// * `file_path` - The path to the PNG file.
///
/// # Returns
///
/// An RGBA8 typed image.
pub fn read_image_png_rgba8(file_path: impl AsRef<Path>) -> Result<Rgba8<CpuAllocator>, IoError> {
    let (buf, size) = read_png_impl(file_path)?;
    Ok(Rgba8::from_size_vec(size.into(), buf, CpuAllocator)?)
}

/// Read a PNG image as RGB16.
///
/// # Arguments
///
/// * `file_path` - The path to the PNG file.
///
/// # Returns
///
/// An RGB16 typed image.
pub fn read_image_png_rgb16(file_path: impl AsRef<Path>) -> Result<Rgb16<CpuAllocator>, IoError> {
    let (buf, size) = read_png_impl(file_path)?;
    let buf_u16 = convert_buf_u8_u16(buf);

    Ok(Rgb16::from_size_vec(size.into(), buf_u16, CpuAllocator)?)
}

/// Read a PNG image as RGBA16.
///
/// # Arguments
///
/// * `file_path` - The path to the PNG file.
///
/// # Returns
///
/// An RGBA16 typed image.
pub fn read_image_png_rgba16(file_path: impl AsRef<Path>) -> Result<Rgba16<CpuAllocator>, IoError> {
    let (buf, size) = read_png_impl(file_path)?;
    let buf_u16 = convert_buf_u8_u16(buf);

    Ok(Rgba16::from_size_vec(size.into(), buf_u16, CpuAllocator)?)
}

/// Read a PNG image as grayscale (Gray16).
///
/// # Arguments
///
/// * `file_path` - The path to the PNG file.
///
/// # Returns
///
/// A Gray16 typed image.
pub fn read_image_png_mono16(file_path: impl AsRef<Path>) -> Result<Gray16<CpuAllocator>, IoError> {
    let (buf, size) = read_png_impl(file_path)?;
    let buf_u16 = convert_buf_u8_u16(buf);

    Ok(Gray16::from_size_vec(size.into(), buf_u16, CpuAllocator)?)
}

/// Decodes a PNG image with as grayscale (Gray8) from Raw Bytes.
///
/// # Arguments
///
/// - `src` - Raw bytes of the png file
/// - `dst` - A mutable reference to your `Gray8` image
pub fn decode_image_png_mono8<A: ImageAllocator>(
    src: &[u8],
    dst: &mut Gray8<A>,
) -> Result<(), IoError> {
    let size = dst.size();
    decode_png_impl::<1>(src, dst.as_slice_mut(), size)
}

/// Decodes a PNG image with a three channel (rgb8) from Raw Bytes.
///
/// # Arguments
///
/// - `src` - Raw bytes of the png file
/// - `dst` - A mutable reference to your `Rgb8` image
pub fn decode_image_png_rgb8<A: ImageAllocator>(
    src: &[u8],
    dst: &mut Rgb8<A>,
) -> Result<(), IoError> {
    let size = dst.size();
    decode_png_impl::<3>(src, dst.as_slice_mut(), size)
}

/// Decodes a PNG image with a four channel (rgba8) from Raw Bytes.
///
/// # Arguments
///
/// - `src` - Raw bytes of the png file
/// - `dst` - A mutable reference to your `Rgba8` image
pub fn decode_image_png_rgba8<A: ImageAllocator>(
    src: &[u8],
    dst: &mut Rgba8<A>,
) -> Result<(), IoError> {
    let size = dst.size();
    decode_png_impl::<4>(src, dst.as_slice_mut(), size)
}

/// Decodes a PNG (16 Bit) image as grayscale (Gray16) from Raw Bytes.
///
/// # Arguments
///
/// - `src` - Raw bytes of the png file
/// - `dst` - A mutable reference to your `Gray16` image
pub fn decode_image_png_mono16<A: ImageAllocator>(
    src: &[u8],
    dst: &mut Gray16<A>,
) -> Result<(), IoError> {
    let mut image_u8 = convert_buf_u16_u8(dst.as_slice());
    decode_png_impl::<1>(src, image_u8.as_mut_slice(), dst.size())?;
    convert_buf_u8_u16_into_slice(image_u8.as_slice(), dst.as_slice_mut());
    Ok(())
}

/// Decodes a PNG (16 Bit) image with a three channel (rgb16) from Raw Bytes.
///
/// # Arguments
///
/// - `src` - Raw bytes of the png file
/// - `dst` - A mutable reference to your `Rgb16` image
pub fn decode_image_png_rgb16<A: ImageAllocator>(
    src: &[u8],
    dst: &mut Rgb16<A>,
) -> Result<(), IoError> {
    let mut image_u8 = convert_buf_u16_u8(dst.as_slice());
    decode_png_impl::<3>(src, image_u8.as_mut_slice(), dst.size())?;
    convert_buf_u8_u16_into_slice(image_u8.as_slice(), dst.as_slice_mut());
    Ok(())
}

/// Decodes a PNG (16 Bit) image with as RGBA (Rgba16) from Raw Bytes.
///
/// # Arguments
///
/// - `src` - Raw bytes of the png file
/// - `dst` - A mutable reference to your `Rgba16` image
pub fn decode_image_png_rgba16<A: ImageAllocator>(
    src: &[u8],
    dst: &mut Rgba16<A>,
) -> Result<(), IoError> {
    let mut image_u8 = convert_buf_u16_u8(dst.as_slice());
    decode_png_impl::<4>(src, image_u8.as_mut_slice(), dst.size())?;
    convert_buf_u8_u16_into_slice(image_u8.as_slice(), dst.as_slice_mut());
    Ok(())
}

/// Decodes PNG image metadata from raw bytes without decoding pixel data.
///
/// # Arguments
///
/// - `src` - Raw bytes of the PNG file
///
/// # Returns
///
/// An `ImageLayout` containing the image metadata (size, channels, pixel format).
pub fn decode_image_png_layout(src: &[u8]) -> Result<ImageLayout, IoError> {
    let cursor = Cursor::new(src);
    let decoder = Decoder::new(cursor);
    let reader = decoder
        .read_info()
        .map_err(|e| IoError::PngDecodeError(e.to_string()))?;

    let info = reader.info();
    let size = ImageSize {
        width: info.width as usize,
        height: info.height as usize,
    };

    let channels: u8 = match info.color_type {
        ColorType::Grayscale => 1,
        ColorType::Rgb => 3,
        ColorType::Rgba => 4,
        ColorType::GrayscaleAlpha => 2,
        ColorType::Indexed => 1,
    };

    let pixel_format = match info.bit_depth {
        BitDepth::Eight => PixelFormat::U8,
        BitDepth::Sixteen => PixelFormat::U16,
        other => {
            return Err(IoError::PngDecodeError(format!(
                "Unsupported bit depth: {:?}",
                other
            )))
        }
    };

    Ok(ImageLayout::new(size, channels, pixel_format))
}

// utility function to read the png file
fn read_png_impl(file_path: impl AsRef<Path>) -> Result<(Vec<u8>, [usize; 2]), IoError> {
    // verify the file exists
    let file_path = file_path.as_ref();
    if !file_path.exists() {
        return Err(IoError::FileDoesNotExist(file_path.to_path_buf()));
    }

    // verify the file extension
    if let Some(extension) = file_path.extension() {
        if extension != "png" {
            return Err(IoError::InvalidFileExtension(file_path.to_path_buf()));
        }
    } else {
        return Err(IoError::InvalidFileExtension(file_path.to_path_buf()));
    }

    let file = fs::File::open(file_path)?;
    let reader = BufReader::new(file);
    let mut reader = Decoder::new(reader)
        .read_info()
        .map_err(|e| IoError::PngDecodeError(e.to_string()))?;

    let buffer_size = reader
        .output_buffer_size()
        .ok_or_else(|| IoError::PngDecodeError("PNG output buffer size overflowed".into()))?;
    let mut buf = vec![0; buffer_size];
    let info = reader
        .next_frame(&mut buf)
        .map_err(|e| IoError::PngDecodeError(e.to_string()))?;

    Ok((buf, [info.width as usize, info.height as usize]))
}

// Utility function to decode png files from raw bytes
fn decode_png_impl<const C: usize>(
    src: &[u8],
    dst: &mut [u8],
    image_size: ImageSize,
) -> Result<(), IoError> {
    let cursor = Cursor::new(src);
    let mut reader = Decoder::new(cursor)
        .read_info()
        .map_err(|e| IoError::PngDecodeError(e.to_string()))?;

    let image_info = reader.info();
    if image_info.size() != (image_size.width as u32, image_size.height as u32) {
        return Err(IoError::DecodeMismatchResolution(
            image_info.height as usize,
            image_info.width as usize,
            image_size.height,
            image_size.width,
        ));
    }

    let buffer_size = reader
        .output_buffer_size()
        .ok_or_else(|| IoError::PngDecodeError("PNG output buffer size overflowed".into()))?;

    if dst.len() < buffer_size {
        return Err(IoError::InvalidBufferSize(dst.len(), buffer_size));
    }

    let _ = reader
        .next_frame(dst)
        .map_err(|e| IoError::PngDecodeError(e.to_string()))?;

    Ok(())
}

/// Writes the given PNG _(rgb8)_ data to the given file path.
///
/// # Arguments
///
/// - `file_path` - The path to the PNG image.
/// - `image` - The Rgb8 image to write.
pub fn write_image_png_rgb8<A: ImageAllocator>(
    file_path: impl AsRef<Path>,
    image: &Image<u8, 3, A>,
) -> Result<(), IoError> {
    write_png_impl(
        file_path,
        image.as_slice(),
        image.size(),
        BitDepth::Eight,
        ColorType::Rgb,
    )
}

/// Writes the given PNG _(rgba8)_ data to the given file path.
///
/// # Arguments
///
/// - `file_path` - The path to the PNG image.
/// - `image` - The Rgba8 image to write.
pub fn write_image_png_rgba8<A: ImageAllocator>(
    file_path: impl AsRef<Path>,
    image: &Image<u8, 4, A>,
) -> Result<(), IoError> {
    write_png_impl(
        file_path,
        image.as_slice(),
        image.size(),
        BitDepth::Eight,
        ColorType::Rgba,
    )
}

/// Writes the given PNG _(grayscale 8-bit)_ data to the given file path.
///
/// # Arguments
///
/// - `file_path` - The path to the PNG image.
/// - `image` - The Gray8 image to write.
pub fn write_image_png_gray8<A: ImageAllocator>(
    file_path: impl AsRef<Path>,
    image: &Image<u8, 1, A>,
) -> Result<(), IoError> {
    write_png_impl(
        file_path,
        image.as_slice(),
        image.size(),
        BitDepth::Eight,
        ColorType::Grayscale,
    )
}

/// Writes the given PNG _(rgb16)_ data to the given file path.
///
/// # Arguments
///
/// - `file_path` - The path to the PNG image.
/// - `image` - The Rgb16 image to write.
pub fn write_image_png_rgb16<A: ImageAllocator>(
    file_path: impl AsRef<Path>,
    image: &Image<u16, 3, A>,
) -> Result<(), IoError> {
    let image_size = image.size();
    let image_buf = convert_buf_u16_u8(image.as_slice());

    write_png_impl(
        file_path,
        &image_buf,
        image_size,
        BitDepth::Sixteen,
        ColorType::Rgb,
    )
}

/// Writes the given PNG _(rgba16)_ data to the given file path.
///
/// # Arguments
///
/// - `file_path` - The path to the PNG image.
/// - `image` - The Rgba16 image to write.
pub fn write_image_png_rgba16<A: ImageAllocator>(
    file_path: impl AsRef<Path>,
    image: &Image<u16, 4, A>,
) -> Result<(), IoError> {
    let image_size = image.size();
    let image_buf = convert_buf_u16_u8(image.as_slice());

    write_png_impl(
        file_path,
        &image_buf,
        image_size,
        BitDepth::Sixteen,
        ColorType::Rgba,
    )
}

/// Writes the given PNG _(grayscale 16-bit)_ data to the given file path.
///
/// # Arguments
///
/// - `file_path` - The path to the PNG image.
/// - `image` - The Gray16 image to write.
pub fn write_image_png_gray16<A: ImageAllocator>(
    file_path: impl AsRef<Path>,
    image: &Image<u16, 1, A>,
) -> Result<(), IoError> {
    let image_size = image.size();
    let image_buf = convert_buf_u16_u8(image.as_slice());

    write_png_impl(
        file_path,
        &image_buf,
        image_size,
        BitDepth::Sixteen,
        ColorType::Grayscale,
    )
}

fn write_png_impl(
    file_path: impl AsRef<Path>,
    image_data: &[u8],
    image_size: ImageSize,
    // Make sure you set `depth` correctly
    depth: BitDepth,
    color_type: ColorType,
) -> Result<(), IoError> {
    let file = File::create(file_path)?;

    let mut encoder = Encoder::new(file, image_size.width as u32, image_size.height as u32);
    encoder.set_color(color_type);
    encoder.set_depth(depth);

    let mut writer = encoder
        .write_header()
        .map_err(|e| IoError::PngEncodingError(e.to_string()))?;
    writer
        .write_image_data(image_data)
        .map_err(|e| IoError::PngEncodingError(e.to_string()))?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::error::IoError;
    use std::fs::{create_dir_all, read};

    #[test]
    fn read_png_mono8() -> Result<(), IoError> {
        let image = read_image_png_mono8("../../tests/data/dog.png")?;
        assert_eq!(image.size().width, 258);
        assert_eq!(image.size().height, 195);
        Ok(())
    }

    #[test]
    fn read_write_png_rgb8() -> Result<(), IoError> {
        let tmp_dir = tempfile::tempdir()?;
        create_dir_all(tmp_dir.path())?;

        let file_path = tmp_dir.path().join("dog-rgb8.png");
        let image_data = read_image_png_rgb8("../../tests/data/dog-rgb8.png")?;
        write_image_png_rgb8(&file_path, &image_data)?;

        let image_data_back = read_image_png_rgb8(&file_path)?;
        assert!(file_path.exists(), "File does not exist: {file_path:?}");

        assert_eq!(image_data_back.cols(), 258);
        assert_eq!(image_data_back.rows(), 195);
        assert_eq!(image_data_back.num_channels(), 3);

        Ok(())
    }

    #[test]
    fn read_write_png_rgb16() -> Result<(), IoError> {
        let tmp_dir = tempfile::tempdir()?;
        create_dir_all(tmp_dir.path())?;

        let file_path = tmp_dir.path().join("rgb16.png");
        let image_data = read_image_png_rgb16("../../tests/data/rgb16.png")?;
        write_image_png_rgb16(&file_path, &image_data)?;

        let image_data_back = read_image_png_rgb16(&file_path)?;
        assert!(file_path.exists(), "File does not exist: {file_path:?}");

        assert_eq!(image_data_back.cols(), 32);
        assert_eq!(image_data_back.rows(), 32);
        assert_eq!(image_data_back.num_channels(), 3);

        Ok(())
    }

    #[test]
    fn decode_png() -> Result<(), IoError> {
        let bytes = read("../../tests/data/dog-rgb8.png")?;
        let mut image = Rgb8::from_size_val([258, 195].into(), 0, CpuAllocator)?;
        decode_image_png_rgb8(&bytes, &mut image)?;

        assert_eq!(image.cols(), 258);
        assert_eq!(image.rows(), 195);
        assert_eq!(image.num_channels(), 3);

        Ok(())
    }
}
