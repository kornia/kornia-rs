use std::{fs::File, path::Path};

use kornia_image::Image;
use png::{Decoder, Encoder, ColorType};

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

/// Write a grayscale image with a single channel (gray8) to a PNG file.
///
/// # Arguments
///
/// * `file_path` - The path to save the PNG file.
/// * `src` - The grayscale image to save.
///
/// # Returns
///
/// `Ok(())` if the image was successfully written, or an error otherwise.
///
/// # Example
///
/// ```
/// use kornia_image::{Image, ImageSize};
/// use kornia_io::png::write_image_png_gray8;
///
/// let image = Image::<u8, 1>::new(
///     ImageSize {
///         width: 2,
///         height: 1,
///     },
///     vec![0, 255],
/// ).unwrap();
///
/// write_image_png_gray8("output.png", image).unwrap();
/// ```
pub fn write_image_png_gray8(file_path: impl AsRef<Path>, src: Image<u8, 1>) -> Result<(), IoError> {
    let file_path = file_path.as_ref();
    
    // Create the output file
    let file = File::create(file_path)?;
    
    let width = src.width() as u32;
    let height = src.height() as u32;
    
    // Create PNG encoder
    let mut encoder = Encoder::new(file, width, height);
    encoder.set_color(ColorType::Grayscale);
    encoder.set_depth(png::BitDepth::Eight);
    
    let mut writer = encoder.write_header()
        .map_err(|e| IoError::PngDecodeError(e.to_string()))?;
    
    // Write the image data
    writer.write_image_data(src.as_slice())
        .map_err(|e| IoError::PngDecodeError(e.to_string()))?;
    
    Ok(())
}

/// Write a RGB image with three channels (rgb8) to a PNG file.
///
/// # Arguments
///
/// * `file_path` - The path to save the PNG file.
/// * `src` - The RGB image to save.
///
/// # Returns
///
/// `Ok(())` if the image was successfully written, or an error otherwise.
///
/// # Example
///
/// ```
/// use kornia_image::{Image, ImageSize};
/// use kornia_io::png::write_image_png_rgb8;
///
/// let image = Image::<u8, 3>::new(
///     ImageSize {
///         width: 2,
///         height: 1,
///     },
///     vec![255, 0, 0, 0, 255, 0],
/// ).unwrap();
///
/// write_image_png_rgb8("output.png", image).unwrap();
/// ```
pub fn write_image_png_rgb8(file_path: impl AsRef<Path>, src: Image<u8, 3>) -> Result<(), IoError> {
    let file_path = file_path.as_ref();
    
    // Create the output file
    let file = File::create(file_path)?;
    
    let width = src.width() as u32;
    let height = src.height() as u32;
    
    // Create PNG encoder
    let mut encoder = Encoder::new(file, width, height);
    encoder.set_color(ColorType::Rgb);
    encoder.set_depth(png::BitDepth::Eight);
    
    let mut writer = encoder.write_header()
        .map_err(|e| IoError::PngDecodeError(e.to_string()))?;
    
    // Write the image data
    writer.write_image_data(src.as_slice())
        .map_err(|e| IoError::PngDecodeError(e.to_string()))?;
    
    Ok(())
}

/// Write a RGBA image with four channels (rgba8) to a PNG file.
///
/// # Arguments
///
/// * `file_path` - The path to save the PNG file.
/// * `src` - The RGBA image to save.
///
/// # Returns
///
/// `Ok(())` if the image was successfully written, or an error otherwise.
///
/// # Example
///
/// ```
/// use kornia_image::{Image, ImageSize};
/// use kornia_io::png::write_image_png_rgba8;
///
/// let image = Image::<u8, 4>::new(
///     ImageSize {
///         width: 2,
///         height: 1,
///     },
///     vec![255, 0, 0, 255, 0, 255, 0, 128],
/// ).unwrap();
///
/// write_image_png_rgba8("output.png", image).unwrap();
/// ```
pub fn write_image_png_rgba8(file_path: impl AsRef<Path>, src: Image<u8, 4>) -> Result<(), IoError> {
    let file_path = file_path.as_ref();
    
    // Create the output file
    let file = File::create(file_path)?;
    
    let width = src.width() as u32;
    let height = src.height() as u32;
    
    // Create PNG encoder
    let mut encoder = Encoder::new(file, width, height);
    encoder.set_color(ColorType::Rgba);
    encoder.set_depth(png::BitDepth::Eight);
    
    let mut writer = encoder.write_header()
        .map_err(|e| IoError::PngDecodeError(e.to_string()))?;
    
    // Write the image data
    writer.write_image_data(src.as_slice())
        .map_err(|e| IoError::PngDecodeError(e.to_string()))?;
    
    Ok(())
}

/// Write a grayscale image with a single channel (gray16) to a PNG file.
///
/// # Arguments
///
/// * `file_path` - The path to save the PNG file.
/// * `src` - The grayscale image to save.
///
/// # Returns
///
/// `Ok(())` if the image was successfully written, or an error otherwise.
///
/// # Example
///
/// ```
/// use kornia_image::{Image, ImageSize};
/// use kornia_io::png::write_image_png_gray16;
///
/// let image = Image::<u16, 1>::new(
///     ImageSize {
///         width: 2,
///         height: 1,
///     },
///     vec![0, 65535],
/// ).unwrap();
///
/// write_image_png_gray16("output.png", image).unwrap();
/// ```
pub fn write_image_png_gray16(file_path: impl AsRef<Path>, src: Image<u16, 1>) -> Result<(), IoError> {
    let file_path = file_path.as_ref();
    
    // Create the output file
    let file = File::create(file_path)?;
    
    let width = src.width() as u32;
    let height = src.height() as u32;
    
    // Create PNG encoder
    let mut encoder = Encoder::new(file, width, height);
    encoder.set_color(ColorType::Grayscale);
    encoder.set_depth(png::BitDepth::Sixteen);
    
    // Convert u16 data to big-endian byte representation
    let mut bytes = Vec::with_capacity(src.as_slice().len() * 2);
    for &pixel in src.as_slice() {
        let be_bytes = pixel.to_be_bytes();
        bytes.extend_from_slice(&be_bytes);
    }
    
    let mut writer = encoder.write_header()
        .map_err(|e| IoError::PngDecodeError(e.to_string()))?;
    
    // Write the image data
    writer.write_image_data(&bytes)
        .map_err(|e| IoError::PngDecodeError(e.to_string()))?;
    
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

    #[test]
    fn write_read_png_gray8() -> Result<(), IoError> {
        use kornia_image::{Image, ImageSize};
        use std::path::PathBuf;
        use tempfile::tempdir;
        use crate::png::{write_image_png_gray8, read_image_png_mono8};
        
        // Create a temporary directory for our test file
        let temp_dir = tempdir()?;
        let file_path = temp_dir.path().join("test_gray8.png");
        
        // Create a test image
        let image = Image::<u8, 1>::new(
            ImageSize {
                width: 2,
                height: 2,
            },
            vec![0, 255, 128, 64],
        )?;
        
        // Write the image to a file
        write_image_png_gray8(&file_path, image.clone())?;
        
        // Read the image back
        let read_image = read_image_png_mono8(&file_path)?;
        
        // Check that the images match
        assert_eq!(read_image.size(), image.size());
        assert_eq!(read_image.as_slice(), image.as_slice());
        
        Ok(())
    }

    #[test]
    fn write_read_png_rgb8() -> Result<(), IoError> {
        use kornia_image::{Image, ImageSize};
        use tempfile::tempdir;
        use crate::png::{write_image_png_rgb8, read_image_png_rgb8};
        
        // Create a temporary directory for our test file
        let temp_dir = tempdir()?;
        let file_path = temp_dir.path().join("test_rgb8.png");
        
        // Create a test image
        let image = Image::<u8, 3>::new(
            ImageSize {
                width: 2,
                height: 2,
            },
            vec![
                255, 0, 0,  // Red
                0, 255, 0,  // Green
                0, 0, 255,  // Blue
                255, 255, 0 // Yellow
            ],
        )?;
        
        // Write the image to a file
        write_image_png_rgb8(&file_path, image.clone())?;
        
        // Read the image back
        let read_image = read_image_png_rgb8(&file_path)?;
        
        // Check that the images match
        assert_eq!(read_image.size(), image.size());
        assert_eq!(read_image.as_slice(), image.as_slice());
        
        Ok(())
    }

    #[test]
    fn write_read_png_rgba8() -> Result<(), IoError> {
        use kornia_image::{Image, ImageSize};
        use tempfile::tempdir;
        use crate::png::{write_image_png_rgba8, read_image_png_rgba8};
        
        // Create a temporary directory for our test file
        let temp_dir = tempdir()?;
        let file_path = temp_dir.path().join("test_rgba8.png");
        
        // Create a test image
        let image = Image::<u8, 4>::new(
            ImageSize {
                width: 2,
                height: 1,
            },
            vec![
                255, 0, 0, 255,    // Red with full alpha
                0, 255, 0, 128     // Green with half alpha
            ],
        )?;
        
        // Write the image to a file
        write_image_png_rgba8(&file_path, image.clone())?;
        
        // Read the image back
        let read_image = read_image_png_rgba8(&file_path)?;
        
        // Check that the images match
        assert_eq!(read_image.size(), image.size());
        assert_eq!(read_image.as_slice(), image.as_slice());
        
        Ok(())
    }

    #[test]
    fn write_read_png_gray16() -> Result<(), IoError> {
        use kornia_image::{Image, ImageSize};
        use tempfile::tempdir;
        use crate::png::{write_image_png_gray16, read_image_png_mono16};
        
        // Create a temporary directory for our test file
        let temp_dir = tempdir()?;
        let file_path = temp_dir.path().join("test_gray16.png");
        
        // Create a test image
        let image = Image::<u16, 1>::new(
            ImageSize {
                width: 2,
                height: 2,
            },
            vec![0, 65535, 32768, 16384],
        )?;
        
        // Write the image to a file
        write_image_png_gray16(&file_path, image.clone())?;
        
        // Read the image back
        let read_image = read_image_png_mono16(&file_path)?;
        
        // Check that the images match
        assert_eq!(read_image.size(), image.size());
        assert_eq!(read_image.as_slice(), image.as_slice());
        
        Ok(())
    }
}
