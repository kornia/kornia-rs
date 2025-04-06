use crate::error::IoError;

use kornia_image::{Image, ImageSize};
use tiff::decoder::Decoder;
use tiff::encoder::{
    colortype::{ColorType,RGB8,Gray8}, 
    TiffEncoder,
    TiffValue,
};
use tiff::TiffResult;
use std::fs::File;
use std::path::Path;
use std::io::BufWriter;

/// Writes the given TIFF (RGB8) data to the specified file path.
///
/// # Arguments
///
/// * `file_path` - The path to the TIFF file to write.
/// * `image` - The image tensor containing TIFF image data (3 channels).
pub fn write_image_tiff_rgb8(
    file_path: impl AsRef<Path>,
    image: &Image<u8, 3>,
) -> Result<(), IoError> {
    write_image_tiff_internal::<RGB8,3>(file_path, image)
}

/// Writes the given TIFF (Grayscale) data to the specified file path.
/// 
/// # Arguments
/// 
/// * `file_path` - The path to the TIFF file to write.
/// * `image` - The image tensor containing TIFF image data (1 channel).
pub fn write_image_tiff_gray8(
    file_path: impl AsRef<Path>,
    image: &Image<u8, 1>,
) -> Result<(), IoError> {
    write_image_tiff_internal::<Gray8,1>(file_path, image)
}


/// Writes the given TIFF (Grayscale) data to the specified file path.
///
/// # Arguments
///
/// * `file_path` - The path to the TIFF file to write.
/// * `image` - The image tensor containing TIFF image data (1 channel).
fn write_image_tiff_internal<T, const C:usize>(
    file_path: impl AsRef<Path>,
    image: &Image<u8, C>,
) -> Result<(), IoError>
where 
    T: ColorType<Inner = u8>,  // Đảm bảo T::Inner là u8
    [T::Inner]:TiffValue,   
{
    if image.num_channels() != C {
        return Err(IoError::TiffEncodingError(
            format!("Image has {} channels, but expected {}", image.num_channels(), C)
        ));

    }
    
    let file = File::create(file_path.as_ref())?;
    let writer = BufWriter::new(file);
    let mut tiff = TiffEncoder::new(writer)?;
    let image_encoder = tiff.new_image::<T>(image.width() as u32, image.height() as u32)?;
    image_encoder.write_data(image.as_slice())?;

    Ok(())

}

/// Reads a TIFF image with a single channel (rgb8).
/// #Arguments
/// file_path - The path to the TIFF file.
/// 
pub fn read_image_tiff_rgb8(
    file_path: impl AsRef<Path>,
) -> Result<Image<u8, 3>, IoError> {
    read_image_tiff_internal::<3>(file_path)
}

/// Reads a TIFF image with a single channel (gray8).
/// #Arguments
/// file_path - The path to the TIFF file.
pub fn read_image_tiff_gray8(
    file_path: impl AsRef<Path>,
) -> Result<Image<u8, 1>, IoError> {
    read_image_tiff_internal::<1>(file_path)
}


// Read a TIFF image with a single channel (rgb8).
//
fn read_image_tiff_internal<const N: usize>(
    file_path: impl AsRef<Path>,
) -> Result<Image<u8,N>, IoError> {
    let file_path = file_path.as_ref().to_owned();

    if !file_path.exists() {
        return Err(IoError::FileDoesNotExist(file_path.to_path_buf()));
    }

    if file_path.extension().map_or(true, |ext| {
        !ext.eq_ignore_ascii_case("tif") && !ext.eq_ignore_ascii_case("tiff")
    }) {
        return Err(IoError::InvalidFileExtension(file_path.to_path_buf()));
    }

    let file = File::open(&file_path)?;
    let mut decoder = Decoder::new(file).map_err(IoError::TiffError)?;
    // read the image data
    let decoding_result = decoder.read_image().map_err(IoError::TiffError)?;
    let vec_data: Vec<u8> = match decoding_result {
        tiff::decoder::DecodingResult::U8(data) => data,
        _ => return Err(IoError::TiffEncodingError("Unsupported data type".to_string())),
    };
    let (width, height) = decoder.dimensions()?;
    println!("width: {}, height: {}", width, height);
    let image = Image::new(
        ImageSize {
            width: width as usize,
            height: height as usize,
        },
        vec_data,
    )?;

    Ok(image)
}



#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use std::path::PathBuf;
    use kornia_image::ImageSize;
    #[test]
    fn test_write_image_tiff_rgb8() {
        let image = Image::new(ImageSize { width: 100, height: 100 }, vec![0; 30000]).unwrap();
        let file_path = PathBuf::from("test_rgb8.tiff");
        assert!(write_image_tiff_rgb8(&file_path, &image).is_ok());
        fs::remove_file(file_path).unwrap();
    }

    #[test]
    fn test_write_image_tiff_gray8() {
        let image = Image::new(ImageSize { width: 100, height: 100 }, vec![0; 10000]).unwrap();
        let file_path = PathBuf::from("test_gray8.tiff");
        assert!(write_image_tiff_gray8(&file_path, &image).is_ok());
        fs::remove_file(file_path).unwrap();
    }

    #[test]
    fn test_read_image_tiff_rgb8() {
        let file_path = PathBuf::from("../../tests/data/example.tiff");
        let image = read_image_tiff_internal::<3>(&file_path).unwrap();
        assert_eq!(image.width(), 1400);
        assert_eq!(image.height(), 934);
    }   
}