use crate::error::IoError;
use image_webp::WebPDecoder;
use kornia_image::{
    allocator::{CpuAllocator, ImageAllocator},
    color_spaces::{Gray8, Rgb8, Rgba8},
    Image, ImageLayout, ImageSize, PixelFormat,
};

use std::{fs, io::{BufReader, Cursor}, path::Path};


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


#[cfg(test)]
mod tests {
    use super::*;
    use std::fs::{create_dir_all, read};

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
}