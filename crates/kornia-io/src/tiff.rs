use crate::error::IoError;

use kornia_image::{Image, ImageSize};
use tiff::decoder::{Decoder,DecodingResult};
use tiff::encoder::{
    colortype::{ColorType,RGB8,RGBA8,Gray8}, 
    TiffEncoder,
    TiffValue,
};
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

/// Writes the given TIFF (RGBA8) data to the specified file path.
/// 
/// # Arguments
/// 
/// * `file_path` - The path to the TIFF file to write.
/// * * `image` - The image tensor containing TIFF image data (4 channels).
pub fn write_image_tiff_rgba8(
    file_path: impl AsRef<Path>,
    image: &Image<u8, 4>,
) -> Result<(), IoError> {
    write_image_tiff_internal::<RGBA8,4>(file_path, image)
}


/// Writes the given TIFF (Grayscale) data to the specified file path.
///
/// # Arguments
///
/// * `file_path` - The path to the TIFF file to write.
/// * `image` - The image tensor containing TIFF image data (1 channel).
fn write_image_tiff_internal<T, const C:usize>(
    file_path: impl AsRef<Path>,
    image: &Image<T::Inner, C>,
) -> Result<(), IoError>
where 
    T: ColorType,
    [T::Inner]:TiffValue,   
{ 
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
    read_image_tiff_internal::<u8, 3>(file_path)
}

/// Reads a TIFF image with a single channel (gray8).
/// #Arguments
/// file_path - The path to the TIFF file.
pub fn read_image_tiff_gray8(
    file_path: impl AsRef<Path>,
) -> Result<Image<u8, 1>, IoError> {
    read_image_tiff_internal::<u8,1>(file_path)
}

/// Reads a TIFF image with RGBA8 channels
/// #Arguments
/// file_path - The path to the TIFF file.
pub fn read_image_tiff_rgba8(
    file_path: impl AsRef<Path>,
) -> Result<Image<u8, 4>, IoError> {
    read_image_tiff_internal::<u8,4>(file_path)
}

/// A trait to convert TIFF values to a specific type.
/// 
/// This trait is used to convert TIFF values (u8 or f32) to a specific type (e.g., u8 or f32).
/// It provides two methods:
pub trait TiffValueConverter: Sized {
    /// * `from_u8`: Converts a u8 value to the specific type.
    fn from_u8(value: u8) -> Self;
    /// * `from_f32`: Converts a f32 value to the specific type.
    fn from_f32(value: f32) -> Self;
}

impl TiffValueConverter for u8 {
    fn from_u8(value: u8) -> Self { value }
    fn from_f32(value: f32) -> Self { (value * 255.0) as u8 }
}

impl TiffValueConverter for f32 {
    fn from_u8(value: u8) -> Self { (value as f32) / 255.0 }
    fn from_f32(value: f32) -> Self { value }
}



// Read a TIFF image with a single channel (rgb8).
//
fn read_image_tiff_internal<T:Clone,const N: usize>(
    file_path: impl AsRef<Path>,
) -> Result<Image<T,N>, IoError> 
    where 
        T: TiffValueConverter,
{
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
    let vec_data:Vec<T> = match decoding_result {
        DecodingResult::U8(data) => data.iter().map(|&x| T::from_u8(x)).collect(),
        DecodingResult::F32(data) => data.iter().map(|&x| T::from_f32(x)).collect(),
        _ => return Err(IoError::TiffEncodingError("Unsupported data type".to_string())),
    };
    let (width, height) = decoder.dimensions()?;
    // check the number of channels
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
        use std::fs;

        let image = Image::new(ImageSize { width: 100, height: 100 }, vec![0; 10000]).unwrap();
        let tmp_dir = tempfile::tempdir().expect("Failed to create temp directory");
        fs::create_dir_all(tmp_dir.path()).expect("Failed to create directory");

        let file_path = tmp_dir.path().join("example.tiff");
        assert!(write_image_tiff_gray8(&file_path, &image).is_ok(), "Failed to write TIFF file");
        assert!(file_path.exists(), "TIFF file was not created");

        let file_size = fs::metadata(&file_path).unwrap().len();
        assert!(file_size > 0, "TIFF file is empty");
    }

    #[test]
    fn test_read_image_tiff_rgba8() {
        let file_path = PathBuf::from("../../tests/data/example.tiff");
        let image = read_image_tiff_rgba8(&file_path).unwrap();
        assert_eq!(image.width(), 1400);
        assert_eq!(image.height(), 934);
    }   

    #[test]
    fn read_write_tiff() -> Result<(), IoError> {
        let tmp_dir = tempfile::tempdir()?;
        fs::create_dir_all(tmp_dir.path())?;

        let file_path = tmp_dir.path().join("example.tiff");
        let image_data = read_image_tiff_rgba8("../../tests/data/example.tiff")?;
        write_image_tiff_rgba8(&file_path, &image_data)?;

        let image_data_back = read_image_tiff_rgba8(&file_path)?;
        assert!(file_path.exists(), "File does not exist: {:?}", file_path);

        assert_eq!(image_data_back.cols(), 1400);
        assert_eq!(image_data_back.rows(), 934);
        assert_eq!(image_data_back.num_channels(), 4);

        Ok(())
    }
}