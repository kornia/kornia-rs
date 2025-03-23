use std::{fs::File, path::Path};

use kornia_image::Image;
use png::Decoder;

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

#[cfg(test)]
mod tests {
    use crate::error::IoError;
    use crate::png::read_image_png_mono8;

    #[test]
    fn read_png_mono8() -> Result<(), IoError> {
        let image = read_image_png_mono8("../../tests/data/dog.png")?;
        assert_eq!(image.size().width, 258);
        assert_eq!(image.size().height, 195);
        Ok(())
    }
}
