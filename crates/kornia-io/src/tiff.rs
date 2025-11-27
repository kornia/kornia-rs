use crate::error::IoError;
use kornia_image::{
    allocator::{CpuAllocator, ImageAllocator},
    color_spaces::{Gray16, Gray8, Grayf32, Rgb16, Rgb8, Rgbf32},
    Image, ImageLayout, PixelFormat, ImageSize,
};
use std::{fs, io::Cursor, path::Path};
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
/// The RGB8 typed image.
pub fn read_image_tiff_rgb8(file_path: impl AsRef<Path>) -> Result<Rgb8<CpuAllocator>, IoError> {
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

    Ok(Rgb8::from_size_vec(size.into(), data, CpuAllocator)?)
}

/// Read a TIFF image and return it as a grayscale image.
///
/// # Arguments
///
/// * `file_path` - The path to the TIFF image.
///
/// # Returns
///
/// The Gray8 typed image.
pub fn read_image_tiff_mono8(file_path: impl AsRef<Path>) -> Result<Gray8<CpuAllocator>, IoError> {
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

    Ok(Gray8::from_size_vec(size.into(), data, CpuAllocator)?)
}

/// Read a TIFF image and return it as a RGB16 image.
///
/// # Arguments
///
/// * `file_path` - The path to the TIFF image.
///
/// # Returns
///
/// The RGB16 typed image.
pub fn read_image_tiff_rgb16(file_path: impl AsRef<Path>) -> Result<Rgb16<CpuAllocator>, IoError> {
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

    Ok(Rgb16::from_size_vec(size.into(), data, CpuAllocator)?)
}

/// Read a TIFF image and return it as a grayscale 16-bit image.
///
/// # Arguments
///
/// * `file_path` - The path to the TIFF image.
///
/// # Returns
///
/// The Gray16 typed image.
pub fn read_image_tiff_mono16(
    file_path: impl AsRef<Path>,
) -> Result<Gray16<CpuAllocator>, IoError> {
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

    Ok(Gray16::from_size_vec(size.into(), data, CpuAllocator)?)
}

/// Read a TIFF image and return it as single precision floating point grayscale image.
///
/// # Arguments
///
/// * `file_path` - The path to the TIFF image.
///
/// # Returns
///
/// The Grayf32 typed image.
pub fn read_image_tiff_mono32f(
    file_path: impl AsRef<Path>,
) -> Result<Grayf32<CpuAllocator>, IoError> {
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

    Ok(Grayf32::from_size_vec(size.into(), data, CpuAllocator)?)
}

/// Read a TIFF image and return it as single precision floating point RGB image.
///
/// # Arguments
///
/// * `file_path` - The path to the TIFF image.
///
/// # Returns
///
/// The Rgbf32 typed image.
pub fn read_image_tiff_rgb32f(
    file_path: impl AsRef<Path>,
) -> Result<Rgbf32<CpuAllocator>, IoError> {
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

    Ok(Rgbf32::from_size_vec(size.into(), data, CpuAllocator)?)
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

fn extract_channels_from_tiff_colortype(colortype: &tiff::ColorType) -> Option<u8> {
    match colortype {
        tiff::ColorType::Gray(_) => Some(1),
        tiff::ColorType::RGB(_) => Some(3),
        tiff::ColorType::Palette(_) => None,
        tiff::ColorType::GrayA(_) => Some(2),
        tiff::ColorType::RGBA(_) => Some(4),
        _ => None,
    }
}

fn pixel_format_from_bits(bits: u8) -> Option<PixelFormat> {
    match bits {
        8 => Some(PixelFormat::U8),
        16 => Some(PixelFormat::U16),
        32 => Some(PixelFormat::F32),
        _ => None,
    }
}

fn pixel_format_from_tiff_colortype(colortype: &tiff::ColorType) -> Option<PixelFormat> {
    match colortype {
        tiff::ColorType::Gray(bits)
        | tiff::ColorType::RGB(bits)
        | tiff::ColorType::Palette(bits)
        | tiff::ColorType::GrayA(bits)
        | tiff::ColorType::RGBA(bits) => pixel_format_from_bits(*bits),
        _ => None,
    }
}

/// Decodes TIFF image metadata from raw bytes without decoding pixel data.
///
/// # Arguments
///
/// - `src` - Raw bytes of the TIFF file
///
/// # Returns
///
/// An `ImageLayout` containing the image metadata (size, channels, pixel format).
pub fn decode_image_tiff_layout(src: &[u8]) -> Result<ImageLayout, IoError> {
    let cursor = Cursor::new(src);
    let mut decoder = tiff::decoder::Decoder::new(cursor)?;

    let (width, height) = decoder.dimensions()?;
    let size = ImageSize {
        width: width as usize,
        height: height as usize,
    };

    let colortype = decoder.colortype()?;
    let num_channels = extract_channels_from_tiff_colortype(&colortype).ok_or_else(|| {
        IoError::TiffDecodingError(tiff::TiffError::UnsupportedError(
            tiff::TiffUnsupportedError::UnknownInterpretation,
        ))
    })?;

    let pixel_format = pixel_format_from_tiff_colortype(&colortype).ok_or_else(|| {
        IoError::TiffDecodingError(tiff::TiffError::UnsupportedError(
            tiff::TiffUnsupportedError::UnknownInterpretation,
        ))
    })?;

    Ok(ImageLayout::new(size, num_channels, pixel_format))
}

/// Decodes TIFF image metadata from raw bytes without decoding pixel data.
///
/// # Deprecated
///
/// Use [`decode_image_tiff_layout`] instead.
#[deprecated(note = "Use decode_image_tiff_layout instead")]
pub fn decode_image_tiff_info(src: &[u8]) -> Result<ImageLayout, IoError> {
    decode_image_tiff_layout(src)
}

/// Reads TIFF image with decoded data and metadata.
pub fn read_image_tiff_with_metadata(
    file_path: impl AsRef<Path>,
) -> Result<(DecodingResult, ImageLayout), IoError> {
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

    let (width, height) = decoder.dimensions()?;
    let size = ImageSize {
        width: width as usize,
        height: height as usize,
    };

    let colortype = decoder.colortype().ok();
    let num_channels_from_metadata = colortype
        .as_ref()
        .and_then(|ct| extract_channels_from_tiff_colortype(ct));

    let result = decoder.read_image()?;

    let pixel_format = match &result {
        DecodingResult::U8(_) => PixelFormat::U8,
        DecodingResult::U16(_) => PixelFormat::U16,
        DecodingResult::F32(_) => PixelFormat::F32,
        _ => {
            return Err(IoError::TiffDecodingError(
                tiff::TiffError::UnsupportedError(
                    tiff::TiffUnsupportedError::UnknownInterpretation,
                ),
            ))
        }
    };
    
    let num_channels = if let Some(channels) = num_channels_from_metadata {
        channels
    } else {
        match &result {
            DecodingResult::U8(data) => (data.len() / (size.width * size.height)) as u8,
            DecodingResult::U16(data) => (data.len() / (size.width * size.height)) as u8,
            DecodingResult::F32(data) => (data.len() / (size.width * size.height)) as u8,
            _ => {
                return Err(IoError::TiffDecodingError(
                    tiff::TiffError::UnsupportedError(
                        tiff::TiffUnsupportedError::UnknownInterpretation,
                    ),
                ))
            }
        }
    };

    let layout = ImageLayout::new(size, num_channels, pixel_format);

    Ok((result, layout))
}

/// Write a TIFF image with a RGB8 color type.
///
/// # Arguments
///
/// * `file_path` - The path to the TIFF image.
/// * `image` - The Rgb8 image to write.
pub fn write_image_tiff_rgb8<A: ImageAllocator>(
    file_path: impl AsRef<Path>,
    image: &Image<u8, 3, A>,
) -> Result<(), IoError> {
    write_image_tiff_impl::<colortype::RGB8, u8>(file_path, image.as_slice(), image.size())
}

/// Write a TIFF image with a mono8 color type.
///
/// # Arguments
///
/// * `file_path` - The path to the TIFF image.
/// * `image` - The Gray8 image to write.
pub fn write_image_tiff_mono8<A: ImageAllocator>(
    file_path: impl AsRef<Path>,
    image: &Image<u8, 1, A>,
) -> Result<(), IoError> {
    write_image_tiff_impl::<colortype::Gray8, u8>(file_path, image.as_slice(), image.size())
}

/// Write a TIFF image with a RGB16 color type.
///
/// # Arguments
///
/// * `file_path` - The path to the TIFF image.
/// * `image` - The Rgb16 image to write.
pub fn write_image_tiff_rgb16<A: ImageAllocator>(
    file_path: impl AsRef<Path>,
    image: &Image<u16, 3, A>,
) -> Result<(), IoError> {
    write_image_tiff_impl::<colortype::RGB16, u16>(file_path, image.as_slice(), image.size())
}

/// Write a TIFF image with a mono16 color type.
///
/// # Arguments
///
/// * `file_path` - The path to the TIFF image.
/// * `image` - The Gray16 image to write.
pub fn write_image_tiff_mono16<A: ImageAllocator>(
    file_path: impl AsRef<Path>,
    image: &Image<u16, 1, A>,
) -> Result<(), IoError> {
    write_image_tiff_impl::<colortype::Gray16, u16>(file_path, image.as_slice(), image.size())
}

/// Write a TIFF image with a single precision as one channel image.
///
/// # Arguments
///
/// * `file_path` - The path to the TIFF image.
/// * `image` - The Grayf32 image to write.
pub fn write_image_tiff_mono32f<A: ImageAllocator>(
    file_path: impl AsRef<Path>,
    image: &Image<f32, 1, A>,
) -> Result<(), IoError> {
    write_image_tiff_impl::<colortype::Gray32Float, f32>(file_path, image.as_slice(), image.size())
}

/// Write a TIFF image with a single precision as three channel image.
///
/// # Arguments
///
/// * `file_path` - The path to the TIFF image.
/// * `image` - The Rgbf32 image to write.
pub fn write_image_tiff_rgb32f<A: ImageAllocator>(
    file_path: impl AsRef<Path>,
    image: &Image<f32, 3, A>,
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
    use std::fs::create_dir_all;

    #[test]
    fn synthetic_write_tiff_rgb8() -> Result<(), IoError> {
        let tmp_dir = tempfile::tempdir()?;
        create_dir_all(tmp_dir.path())?;

        let width = 3;
        let height = 4;
        let channels = 3;

        let data = (0..(width * height * channels)).collect::<Vec<_>>();

        let img_rgb8 = Rgb8::from_size_vec(
            ImageSize {
                width: width as usize,
                height: height as usize,
            },
            data,
            CpuAllocator,
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

        let img_mono8 = Gray8::from_size_vec(
            ImageSize {
                width: width as usize,
                height: height as usize,
            },
            data,
            CpuAllocator,
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

        let img_rgb16 = Rgb16::from_size_vec(
            ImageSize {
                width: width as usize,
                height: height as usize,
            },
            data,
            CpuAllocator,
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

        let img_mono16 = Gray16::from_size_vec(
            ImageSize {
                width: width as usize,
                height: height as usize,
            },
            data,
            CpuAllocator,
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

        let img_mono32f = Grayf32::from_size_vec(
            ImageSize {
                width: width as usize,
                height: height as usize,
            },
            data,
            CpuAllocator,
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

        let img_rgb32f = Rgbf32::from_size_vec(
            ImageSize {
                width: width as usize,
                height: height as usize,
            },
            data,
            CpuAllocator,
        )?;

        let file_path = tmp_dir.path().join("rgb32f.tiff");
        write_image_tiff_rgb32f(&file_path, &img_rgb32f)?;

        let img_rgb32f_back = read_image_tiff_rgb32f(&file_path)?;
        assert_eq!(img_rgb32f_back.as_slice(), img_rgb32f.as_slice());

        Ok(())
    }
}
