use std::{fs, fs::File, path::Path};

use kornia_image::{Image, ImageSize};
use png::{BitDepth, ColorType, Decoder, Encoder};

use crate::{
    convert_buf_u16_u8, convert_buf_u8_u16, convert_buf_u8_u16_into_slice, error::IoError,
};

/// Read a PNG image with a single channel (mono8).
///
/// # Arguments
///
/// * `file_path` - The path to the PNG file.
///
/// # Returns
///
/// A grayscale image with a single channel (mono8).
pub fn read_image_png_mono8(file_path: impl AsRef<Path>) -> Result<Image<u8, 1>, IoError> {
    let (buf, size) = read_png_impl(file_path)?;
    Ok(Image::new(size.into(), buf)?)
}

/// Read a PNG image with a three channels (rgb8).
///
/// # Arguments
///
/// * `file_path` - The path to the PNG file.
///
/// # Returns
///
/// A RGB image with three channels (rgb8).
pub fn read_image_png_rgb8(file_path: impl AsRef<Path>) -> Result<Image<u8, 3>, IoError> {
    let (buf, size) = read_png_impl(file_path)?;
    Ok(Image::new(size.into(), buf)?)
}

/// Read a PNG image with a four channels (rgba8).
///
/// # Arguments
///
/// * `file_path` - The path to the PNG file.
///
/// # Returns
///
/// A RGBA image with four channels (rgba8).
pub fn read_image_png_rgba8(file_path: impl AsRef<Path>) -> Result<Image<u8, 4>, IoError> {
    let (buf, size) = read_png_impl(file_path)?;
    Ok(Image::new(size.into(), buf)?)
}

/// Read a PNG image with a three channels (rgb16).
///
/// # Arguments
///
/// * `file_path` - The path to the PNG file.
///
/// # Returns
///
/// A RGB image with three channels (rgb16).
pub fn read_image_png_rgb16(file_path: impl AsRef<Path>) -> Result<Image<u16, 3>, IoError> {
    let (buf, size) = read_png_impl(file_path)?;
    let buf_u16 = convert_buf_u8_u16(buf);

    Ok(Image::new(size.into(), buf_u16)?)
}

/// Read a PNG image with a four channels (rgba16).
///
/// # Arguments
///
/// * `file_path` - The path to the PNG file.
///
/// # Returns
///
/// A RGB image with four channels (rgb16).
pub fn read_image_png_rgba16(file_path: impl AsRef<Path>) -> Result<Image<u16, 4>, IoError> {
    let (buf, size) = read_png_impl(file_path)?;
    let buf_u16 = convert_buf_u8_u16(buf);

    Ok(Image::new(size.into(), buf_u16)?)
}

/// Read a PNG image with a single channel (mono16).
///
/// # Arguments
///
/// * `file_path` - The path to the PNG file.
///
/// # Returns
///
/// A grayscale image with a single channel (mono16).
pub fn read_image_png_mono16(file_path: impl AsRef<Path>) -> Result<Image<u16, 1>, IoError> {
    let (buf, size) = read_png_impl(file_path)?;
    let buf_u16 = convert_buf_u8_u16(buf);

    Ok(Image::new(size.into(), buf_u16)?)
}

/// Decodes a PNG image with a single channel (mono8) from Raw Bytes.
///
/// # Arguments
///
/// - `image` - A mutable reference to your `Image`
/// - `bytes` - Raw bytes of the png file
pub fn decode_image_png_mono8(src: &[u8], dst: &mut Image<u8, 1>) -> Result<(), IoError> {
    let size = dst.size();
    decode_png_impl::<1>(src, dst.as_slice_mut(), size)
}

/// Decodes a PNG image with a three channel (rgb8) from Raw Bytes.
///
/// # Arguments
///
/// - `image` - A mutable reference to your `Image`
/// - `bytes` - Raw bytes of the png file
pub fn decode_image_png_rgb8(src: &[u8], dst: &mut Image<u8, 3>) -> Result<(), IoError> {
    let size = dst.size();
    decode_png_impl::<3>(src, dst.as_slice_mut(), size)
}

/// Decodes a PNG image with a four channel (rgba8) from Raw Bytes.
///
/// # Arguments
///
/// - `image` - A mutable reference to your `Image`
/// - `bytes` - Raw bytes of the png file
pub fn decode_image_png_rgba8(src: &[u8], dst: &mut Image<u8, 4>) -> Result<(), IoError> {
    let size = dst.size();
    decode_png_impl::<4>(src, dst.as_slice_mut(), size)
}

/// Decodes a PNG (16 Bit) image with a single channel (mono16) from Raw Bytes.
///
/// # Arguments
///
/// - `image` - A mutable reference to your `Image`
/// - `bytes` - Raw bytes of the png file
pub fn decode_image_png_mono16(src: &[u8], dst: &mut Image<u16, 1>) -> Result<(), IoError> {
    let mut image_u8 = convert_buf_u16_u8(dst.as_slice());
    decode_png_impl::<1>(src, image_u8.as_mut_slice(), dst.size())?;
    convert_buf_u8_u16_into_slice(image_u8.as_slice(), dst.as_slice_mut());
    Ok(())
}

/// Decodes a PNG (16 Bit) image with a three channel (rgb16) from Raw Bytes.
///
/// # Arguments
///
/// - `image` - A mutable reference to your `Image`
/// - `bytes` - Raw bytes of the png file
pub fn decode_image_png_rgb16(src: &[u8], dst: &mut Image<u16, 3>) -> Result<(), IoError> {
    let mut image_u8 = convert_buf_u16_u8(dst.as_slice());
    decode_png_impl::<3>(src, image_u8.as_mut_slice(), dst.size())?;
    convert_buf_u8_u16_into_slice(image_u8.as_slice(), dst.as_slice_mut());
    Ok(())
}

/// Decodes a PNG (16 Bit) image with a four channel (rgba16) from Raw Bytes.
///
/// # Arguments
///
/// - `image` - A mutable reference to your `Image`
/// - `bytes` - Raw bytes of the png file
pub fn decode_image_png_rgba16(src: &[u8], dst: &mut Image<u16, 4>) -> Result<(), IoError> {
    let mut image_u8 = convert_buf_u16_u8(dst.as_slice());
    decode_png_impl::<4>(src, image_u8.as_mut_slice(), dst.size())?;
    convert_buf_u8_u16_into_slice(image_u8.as_slice(), dst.as_slice_mut());
    Ok(())
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
    let mut reader = Decoder::new(file)
        .read_info()
        .map_err(|e| IoError::PngDecodeError(e.to_string()))?;

    let mut buf = vec![0; reader.output_buffer_size()];
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
    let mut reader = Decoder::new(src)
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

    if dst.len() < reader.output_buffer_size() {
        return Err(IoError::InvalidBufferSize(
            dst.len(),
            reader.output_buffer_size(),
        ));
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
/// - `image` - The tensor containing the PNG image data.
pub fn write_image_png_rgb8(
    file_path: impl AsRef<Path>,
    image: &Image<u8, 3>,
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
/// - `image` - The tensor containing the PNG image data.
pub fn write_image_png_rgba8(
    file_path: impl AsRef<Path>,
    image: &Image<u8, 4>,
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
/// - `image` - The tensor containing the PNG image data.
pub fn write_image_png_gray8(
    file_path: impl AsRef<Path>,
    image: &Image<u8, 1>,
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
/// - `image` - The tensor containing the PNG image data.
pub fn write_image_png_rgb16(
    file_path: impl AsRef<Path>,
    image: &Image<u16, 3>,
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
/// - `image` - The tensor containing the PNG image data.
pub fn write_image_png_rgba16(
    file_path: impl AsRef<Path>,
    image: &Image<u16, 4>,
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
/// - `image` - The tensor containing the PNG image data.
pub fn write_image_png_gray16(
    file_path: impl AsRef<Path>,
    image: &Image<u16, 1>,
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
        assert!(file_path.exists(), "File does not exist: {:?}", file_path);

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
        assert!(file_path.exists(), "File does not exist: {:?}", file_path);

        assert_eq!(image_data_back.cols(), 32);
        assert_eq!(image_data_back.rows(), 32);
        assert_eq!(image_data_back.num_channels(), 3);

        Ok(())
    }

    #[test]
    fn decode_png() -> Result<(), IoError> {
        let bytes = read("../../tests/data/dog-rgb8.png")?;
        let mut image: Image<u8, 3> = Image::from_size_val([258, 195].into(), 0)?;
        decode_image_png_rgb8(&bytes, &mut image)?;

        assert_eq!(image.cols(), 258);
        assert_eq!(image.rows(), 195);
        assert_eq!(image.num_channels(), 3);

        Ok(())
    }
}
