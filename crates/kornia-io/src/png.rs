use std::{fs::File, path::Path};

use kornia_image::{Image, ImageSize};
use png::{BitDepth, ColorType, Decoder, Encoder};

use crate::error::IoError;

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

    // convert the buffer to u16
    let mut buf_u16 = Vec::with_capacity(buf.len() / 2);
    for chunk in buf.chunks_exact(2) {
        buf_u16.push(u16::from_be_bytes([chunk[0], chunk[1]]));
    }

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

    // convert the buffer to u16
    let mut buf_u16 = Vec::with_capacity(buf.len() / 2);
    for chunk in buf.chunks_exact(2) {
        buf_u16.push(u16::from_be_bytes([chunk[0], chunk[1]]));
    }

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

    // convert the buffer to u16
    let mut buf_u16 = Vec::with_capacity(buf.len() / 2);
    for chunk in buf.chunks_exact(2) {
        buf_u16.push(u16::from_be_bytes([chunk[0], chunk[1]]));
    }

    Ok(Image::new(size.into(), buf_u16)?)
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

    let file = File::open(file_path)?;
    let mut reader = Decoder::new(file)
        .read_info()
        .map_err(|e| IoError::PngDecodeError(e.to_string()))?;

    let mut buf = vec![0; reader.output_buffer_size()];
    let info = reader
        .next_frame(&mut buf)
        .map_err(|e| IoError::PngDecodeError(e.to_string()))?;

    Ok((buf, [info.width as usize, info.height as usize]))
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
    let mut image_buf: Vec<u8> = Vec::with_capacity(image_size.width * image_size.height * 2);

    for buf in image.as_slice() {
        let be_bytes = buf.to_be_bytes();
        image_buf.extend_from_slice(&be_bytes);
    }

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
    let mut image_buf: Vec<u8> = Vec::with_capacity(image_size.width * image_size.height * 2);

    for buf in image.as_slice() {
        let be_bytes = buf.to_be_bytes();
        image_buf.extend_from_slice(&be_bytes);
    }

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
    let mut image_buf: Vec<u8> = Vec::with_capacity(image_size.width * image_size.height * 2);

    for buf in image.as_slice() {
        let bug_be = buf.to_be_bytes();
        image_buf.extend_from_slice(&bug_be);
    }

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
    use std::fs::create_dir_all;

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
}
