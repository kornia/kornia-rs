use crate::error::IoError;
use kornia_image::{Image, ImageSize};
use std::{fs, path::Path};
use tiff::{
    decoder::DecodingResult,
    encoder::{colortype, TiffEncoder},
};

/// Read a TIFF image and return it as an RGB8 image.
///
/// # Arguments
///
/// * `file_path` - The path to the TIFF image.
///
/// # Returns
///
/// The rgb8 image as a `Image<u8, 3>`.
pub fn read_image_tiff_rgb8(file_path: impl AsRef<Path>) -> Result<Image<u8, 3>, IoError> {
    let (result, size) = read_image_tiff_impl(file_path)?;

    let data = match result {
        DecodingResult::U8(data) => data,
        _ => {
            return Err(IoError::TiffDecodingError(
                tiff::TiffError::UnsupportedError(
                    tiff::TiffUnsupportedError::UnknownInterpretation,
                ),
            ))
        }
    };

    Ok(Image::new(size.into(), data)?)
}

/// Read a TIFF image and return it as a mono8 image.
///
/// # Arguments
///
/// * `file_path` - The path to the TIFF image.
///
/// # Returns
///
/// The mono8 image as a `Image<u8, 1>`.
pub fn read_image_tiff_mono8(file_path: impl AsRef<Path>) -> Result<Image<u8, 1>, IoError> {
    let (result, size) = read_image_tiff_impl(file_path)?;

    let data = match result {
        DecodingResult::U8(data) => data,
        _ => {
            return Err(IoError::TiffDecodingError(
                tiff::TiffError::UnsupportedError(
                    tiff::TiffUnsupportedError::UnknownInterpretation,
                ),
            ))
        }
    };

    Ok(Image::new(size.into(), data)?)
}

/// Read a TIFF image and return it as a RGB16 image.
///
/// # Arguments
///
/// * `file_path` - The path to the TIFF image.
///
/// # Returns
///
/// The rgb16 image as a `Image<u16, 3>`.
pub fn read_image_tiff_rgb16(file_path: impl AsRef<Path>) -> Result<Image<u16, 3>, IoError> {
    let (result, size) = read_image_tiff_impl(file_path)?;

    let data = match result {
        DecodingResult::U16(data) => data,
        _ => {
            return Err(IoError::TiffDecodingError(
                tiff::TiffError::UnsupportedError(
                    tiff::TiffUnsupportedError::UnknownInterpretation,
                ),
            ))
        }
    };

    Ok(Image::new(size.into(), data)?)
}

/// Read a TIFF image and return it as a mono16 image.
///
/// # Arguments
///
/// * `file_path` - The path to the TIFF image.
///
/// # Returns
///
/// The mono16 image as a `Image<u16, 1>`.
pub fn read_image_tiff_mono16(file_path: impl AsRef<Path>) -> Result<Image<u16, 1>, IoError> {
    let (result, size) = read_image_tiff_impl(file_path)?;

    let data = match result {
        DecodingResult::U16(data) => data,
        _ => {
            return Err(IoError::TiffDecodingError(
                tiff::TiffError::UnsupportedError(
                    tiff::TiffUnsupportedError::UnknownInterpretation,
                ),
            ))
        }
    };

    Ok(Image::new(size.into(), data)?)
}

/// Read a TIFF image and return it as single precision floating point image with one channel.
///
/// # Arguments
///
/// * `file_path` - The path to the TIFF image.
///
/// # Returns
///
/// The floating point image as a `Image<f32, 1>`.
pub fn read_image_tiff_mono32f(file_path: impl AsRef<Path>) -> Result<Image<f32, 1>, IoError> {
    let (result, size) = read_image_tiff_impl(file_path)?;

    let data = match result {
        DecodingResult::F32(data) => data,
        _ => {
            return Err(IoError::TiffDecodingError(
                tiff::TiffError::UnsupportedError(
                    tiff::TiffUnsupportedError::UnknownInterpretation,
                ),
            ))
        }
    };

    Ok(Image::new(size.into(), data)?)
}

/// Read a TIFF image and return it as single precision floating point image with three channels.
///
/// # Arguments
///
/// * `file_path` - The path to the TIFF image.
///
/// # Returns
///
/// The floating point image as a `Image<f32, 3>`.
pub fn read_image_tiff_rgb32f(file_path: impl AsRef<Path>) -> Result<Image<f32, 3>, IoError> {
    let (result, size) = read_image_tiff_impl(file_path)?;

    let data = match result {
        DecodingResult::F32(data) => data,
        _ => {
            return Err(IoError::TiffDecodingError(
                tiff::TiffError::UnsupportedError(
                    tiff::TiffUnsupportedError::UnknownInterpretation,
                ),
            ))
        }
    };

    Ok(Image::new(size.into(), data)?)
}

fn read_image_tiff_impl(
    file_path: impl AsRef<Path>,
) -> Result<(DecodingResult, [usize; 2]), IoError> {
    let file_path = file_path.as_ref().to_owned();
    if !file_path.exists() {
        return Err(IoError::FileDoesNotExist(file_path.to_path_buf()));
    }

    if file_path.extension().map_or(true, |ext| {
        !ext.eq_ignore_ascii_case("tiff") && !ext.eq_ignore_ascii_case("tif")
    }) {
        return Err(IoError::InvalidFileExtension(file_path.to_path_buf()));
    }

    let tiff_data = fs::File::open(file_path)?;
    let mut decoder = tiff::decoder::Decoder::new(tiff_data)?;

    let result = decoder.read_image()?;
    let (width, height) = decoder.dimensions()?;

    Ok((result, [width as usize, height as usize]))
}

/// Write a TIFF image with a RGB8 color type.
///
/// # Arguments
///
/// * `file_path` - The path to the TIFF image.
/// * `image` - The image to write.
pub fn write_image_tiff_rgb8(
    file_path: impl AsRef<Path>,
    image: &Image<u8, 3>,
) -> Result<(), IoError> {
    write_image_tiff_impl::<colortype::RGB8, u8>(file_path, image.as_slice(), image.size())
}

/// Write a TIFF image with a mono8 color type.
///
/// # Arguments
///
/// * `file_path` - The path to the TIFF image.
/// * `image` - The image to write.
pub fn write_image_tiff_mono8(
    file_path: impl AsRef<Path>,
    image: &Image<u8, 1>,
) -> Result<(), IoError> {
    write_image_tiff_impl::<colortype::Gray8, u8>(file_path, image.as_slice(), image.size())
}

/// Write a TIFF image with a RGB16 color type.
///
/// # Arguments
///
/// * `file_path` - The path to the TIFF image.
/// * `image` - The image to write.
pub fn write_image_tiff_rgb16(
    file_path: impl AsRef<Path>,
    image: &Image<u16, 3>,
) -> Result<(), IoError> {
    write_image_tiff_impl::<colortype::RGB16, u16>(file_path, image.as_slice(), image.size())
}

/// Write a TIFF image with a mono16 color type.
///
/// # Arguments
///
/// * `file_path` - The path to the TIFF image.
/// * `image` - The image to write.
pub fn write_image_tiff_mono16(
    file_path: impl AsRef<Path>,
    image: &Image<u16, 1>,
) -> Result<(), IoError> {
    write_image_tiff_impl::<colortype::Gray16, u16>(file_path, image.as_slice(), image.size())
}

/// Write a TIFF image with a single precision as one channel image.
///
/// # Arguments
///
/// * `file_path` - The path to the TIFF image.
/// * `image` - The image to write.
pub fn write_image_tiff_mono32f(
    file_path: impl AsRef<Path>,
    image: &Image<f32, 1>,
) -> Result<(), IoError> {
    write_image_tiff_impl::<colortype::Gray32Float, f32>(file_path, image.as_slice(), image.size())
}

/// Write a TIFF image with a single precision as three channel image.
///
/// # Arguments
///
/// * `file_path` - The path to the TIFF image.
/// * `image` - The image to write.
pub fn write_image_tiff_rgb32f(
    file_path: impl AsRef<Path>,
    image: &Image<f32, 3>,
) -> Result<(), IoError> {
    write_image_tiff_impl::<colortype::RGB32Float, f32>(file_path, image.as_slice(), image.size())
}

fn write_image_tiff_impl<C, T>(
    file_path: impl AsRef<Path>,
    image_data: &[T],
    image_size: ImageSize,
) -> Result<(), IoError>
where
    C: tiff::encoder::colortype::ColorType<Inner = T>,
    [T]: tiff::encoder::TiffValue,
{
    let file = fs::File::create(file_path)?;

    let mut encoder = TiffEncoder::new(file)?;
    encoder.write_image::<C>(
        image_size.width as u32,
        image_size.height as u32,
        image_data,
    )?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::error::IoError;
    use kornia_image::Image;
    use std::fs::create_dir_all;

    #[test]
    fn synthetic_write_tiff_rgb8() -> Result<(), IoError> {
        let tmp_dir = tempfile::tempdir()?;
        create_dir_all(tmp_dir.path())?;

        let width = 3;
        let height = 4;
        let channels = 3;

        let data = (0..(width * height * channels)).collect::<Vec<_>>();

        let img_rgb8 = Image::<u8, 3>::new(
            ImageSize {
                width: width as usize,
                height: height as usize,
            },
            data,
        )?;

        let file_path = tmp_dir.path().join("rgb8.tiff");
        write_image_tiff_rgb8(&file_path, &img_rgb8)?;

        let img_rgb8_back = read_image_tiff_rgb8(&file_path)?;
        assert_eq!(img_rgb8_back.as_slice(), img_rgb8.as_slice());

        Ok(())
    }

    #[test]
    fn synthetic_write_tiff_mono8() -> Result<(), IoError> {
        let tmp_dir = tempfile::tempdir()?;
        create_dir_all(tmp_dir.path())?;

        let width = 3;
        let height = 4;
        let channels = 1;

        let data = (0..(width * height * channels)).collect::<Vec<_>>();

        let img_mono8 = Image::<u8, 1>::new(
            ImageSize {
                width: width as usize,
                height: height as usize,
            },
            data,
        )?;

        let file_path = tmp_dir.path().join("mono8.tiff");
        write_image_tiff_mono8(&file_path, &img_mono8)?;

        let img_mono8_back = read_image_tiff_mono8(&file_path)?;
        assert_eq!(img_mono8_back.as_slice(), img_mono8.as_slice());

        Ok(())
    }

    #[test]
    fn synthetic_write_tiff_rgb16() -> Result<(), IoError> {
        let tmp_dir = tempfile::tempdir()?;
        create_dir_all(tmp_dir.path())?;

        let width = 3;
        let height = 4;
        let channels = 3;

        let data = (0..(width * height * channels)).collect::<Vec<_>>();

        let img_rgb16 = Image::<u16, 3>::new(
            ImageSize {
                width: width as usize,
                height: height as usize,
            },
            data,
        )?;

        let file_path = tmp_dir.path().join("rgb16.tiff");
        write_image_tiff_rgb16(&file_path, &img_rgb16)?;

        let img_rgb16_back = read_image_tiff_rgb16(&file_path)?;
        assert_eq!(img_rgb16_back.as_slice(), img_rgb16.as_slice());

        Ok(())
    }

    #[test]
    fn synthetic_write_tiff_mono16() -> Result<(), IoError> {
        let tmp_dir = tempfile::tempdir()?;
        create_dir_all(tmp_dir.path())?;

        let width = 3;
        let height = 4;
        let channels = 1;

        let data = (0..(width * height * channels)).collect::<Vec<_>>();

        let img_mono16 = Image::<u16, 1>::new(
            ImageSize {
                width: width as usize,
                height: height as usize,
            },
            data,
        )?;

        let file_path = tmp_dir.path().join("mono16.tiff");
        write_image_tiff_mono16(&file_path, &img_mono16)?;

        let img_mono16_back = read_image_tiff_mono16(&file_path)?;
        assert_eq!(img_mono16_back.as_slice(), img_mono16.as_slice());

        Ok(())
    }

    #[test]
    fn synthetic_write_tiff_monof32() -> Result<(), IoError> {
        let tmp_dir = tempfile::tempdir()?;
        create_dir_all(tmp_dir.path())?;

        let width = 1;
        let height = 2;

        let data = vec![3.0, 2.0];

        let img_mono32f = Image::<f32, 1>::new(
            ImageSize {
                width: width as usize,
                height: height as usize,
            },
            data,
        )?;

        let file_path = tmp_dir.path().join("mono32f.tiff");
        write_image_tiff_mono32f(&file_path, &img_mono32f)?;

        let img_mono32f_back = read_image_tiff_mono32f(&file_path)?;
        assert_eq!(img_mono32f_back.as_slice(), img_mono32f.as_slice());

        Ok(())
    }

    #[test]
    fn synthetic_write_tiff_rgbf32() -> Result<(), IoError> {
        let tmp_dir = tempfile::tempdir()?;
        create_dir_all(tmp_dir.path())?;

        let width = 1;
        let height = 2;

        let data = vec![3.0, 2.0, 1.0, 0.0, 1.0, 2.0];

        let img_rgb32f = Image::<f32, 3>::new(
            ImageSize {
                width: width as usize,
                height: height as usize,
            },
            data,
        )?;

        let file_path = tmp_dir.path().join("rgb32f.tiff");
        write_image_tiff_rgb32f(&file_path, &img_rgb32f)?;

        let img_rgb32f_back = read_image_tiff_rgb32f(&file_path)?;
        assert_eq!(img_rgb32f_back.as_slice(), img_rgb32f.as_slice());

        Ok(())
    }
}
