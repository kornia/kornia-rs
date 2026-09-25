use crate::{error::IoError, limits::check_image_dimensions};
use kornia_image::{
    color_spaces::{Gray16, Gray8, Grayf32, Rgb16, Rgb8, Rgbf32},
    Image, ImageLayout, ImageSize, PixelFormat,
};
use std::{fs, io::Cursor, path::Path};
use tiff::{
    decoder::{DecodingBuffer, DecodingResult, DecodingSampleType},
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
pub fn read_image_tiff_rgb8(file_path: impl AsRef<Path>) -> Result<Rgb8, IoError> {
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

    Ok(Rgb8::from_size_vec(size.into(), data)?)
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
pub fn read_image_tiff_mono8(file_path: impl AsRef<Path>) -> Result<Gray8, IoError> {
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

    Ok(Gray8::from_size_vec(size.into(), data)?)
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
pub fn read_image_tiff_rgb16(file_path: impl AsRef<Path>) -> Result<Rgb16, IoError> {
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

    Ok(Rgb16::from_size_vec(size.into(), data)?)
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
pub fn read_image_tiff_mono16(file_path: impl AsRef<Path>) -> Result<Gray16, IoError> {
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

    Ok(Gray16::from_size_vec(size.into(), data)?)
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
pub fn read_image_tiff_mono32f(file_path: impl AsRef<Path>) -> Result<Grayf32, IoError> {
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

    Ok(Grayf32::from_size_vec(size.into(), data)?)
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
pub fn read_image_tiff_rgb32f(file_path: impl AsRef<Path>) -> Result<Rgbf32, IoError> {
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

    Ok(Rgbf32::from_size_vec(size.into(), data)?)
}

fn read_image_tiff_impl(
    file_path: impl AsRef<Path>,
) -> Result<(DecodingResult, [usize; 2]), IoError> {
    let file_path = file_path.as_ref().to_owned();
    if !file_path.exists() {
        return Err(IoError::FileDoesNotExist(file_path.to_path_buf()));
    }

    if file_path
        .extension()
        .is_none_or(|ext| !ext.eq_ignore_ascii_case("tiff") && !ext.eq_ignore_ascii_case("tif"))
    {
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
    check_image_dimensions(size.width, size.height)?;

    let colortype = decoder.colortype()?;
    let num_channels =
        extract_channels_from_tiff_colortype(&colortype).ok_or(IoError::TiffDecodingError(
            tiff::TiffError::UnsupportedError(tiff::TiffUnsupportedError::UnknownInterpretation),
        ))?;

    let pixel_format =
        pixel_format_from_tiff_colortype(&colortype).ok_or(IoError::TiffDecodingError(
            tiff::TiffError::UnsupportedError(tiff::TiffUnsupportedError::UnknownInterpretation),
        ))?;

    Ok(ImageLayout::new(size, num_channels, pixel_format))
}

/// Decodes a TIFF image with a three channel (rgb8) from Raw Bytes.
///
/// # Arguments
///
/// - `src` - Raw bytes of the TIFF file
/// - `dst` - A mutable reference to your `Rgb8` image
pub fn decode_image_tiff_rgb8(src: &[u8], dst: &mut Rgb8) -> Result<(), IoError> {
    decode_tiff_impl(src, &mut dst.0)
}

/// Decodes a TIFF image with as grayscale (Gray8) from Raw Bytes.
///
/// # Arguments
///
/// - `src` - Raw bytes of the TIFF file
/// - `dst` - A mutable reference to your `Gray8` image
pub fn decode_image_tiff_mono8(src: &[u8], dst: &mut Gray8) -> Result<(), IoError> {
    decode_tiff_impl(src, &mut dst.0)
}

/// Decodes a TIFF (16 Bit) image with a three channel (rgb16) from Raw Bytes.
///
/// # Arguments
///
/// - `src` - Raw bytes of the TIFF file
/// - `dst` - A mutable reference to your `Rgb16` image
pub fn decode_image_tiff_rgb16(src: &[u8], dst: &mut Rgb16) -> Result<(), IoError> {
    decode_tiff_impl(src, &mut dst.0)
}

/// Decodes a TIFF (16 Bit) image as grayscale (Gray16) from Raw Bytes.
///
/// # Arguments
///
/// - `src` - Raw bytes of the TIFF file
/// - `dst` - A mutable reference to your `Gray16` image
pub fn decode_image_tiff_mono16(src: &[u8], dst: &mut Gray16) -> Result<(), IoError> {
    decode_tiff_impl(src, &mut dst.0)
}

/// Decodes a TIFF (32 Bit Float) image as grayscale (Grayf32) from Raw Bytes.
///
/// # Arguments
///
/// - `src` - Raw bytes of the TIFF file
/// - `dst` - A mutable reference to your `Grayf32` image
pub fn decode_image_tiff_mono32f(src: &[u8], dst: &mut Grayf32) -> Result<(), IoError> {
    decode_tiff_impl(src, &mut dst.0)
}

/// Decodes a TIFF (32 Bit Float) image with a three channel (rgbf32) from Raw Bytes.
///
/// # Arguments
///
/// - `src` - Raw bytes of the TIFF file
/// - `dst` - A mutable reference to your `Rgbf32` image
pub fn decode_image_tiff_rgb32f(src: &[u8], dst: &mut Rgbf32) -> Result<(), IoError> {
    decode_tiff_impl(src, &mut dst.0)
}

// Verifies that the TIFF's channel count and sample type match what the caller requested, so
// that e.g. a grayscale or integer TIFF is not silently reinterpreted as RGB or float data.
fn check_tiff_format<R: std::io::Read + std::io::Seek>(
    decoder: &mut tiff::decoder::Decoder<R>,
    expected_channels: u8,
    expected_type: DecodingSampleType,
) -> Result<(), IoError> {
    let colortype = decoder.colortype()?;
    let channels = extract_channels_from_tiff_colortype(&colortype);
    let sample_type = decoder.image_buffer_layout()?.sample_type;
    if channels != Some(expected_channels) || sample_type != Some(expected_type) {
        return Err(IoError::FormatMismatch(format!(
            "TIFF is {colortype:?} ({sample_type:?}), expected {expected_channels} channel(s) of {expected_type:?}"
        )));
    }
    Ok(())
}

// A TIFF sample type that can be decoded directly into a typed pixel buffer.
trait TiffSample: Sized {
    const SAMPLE_TYPE: DecodingSampleType;
    fn as_buffer(buf: &mut [Self]) -> DecodingBuffer<'_>;
}

macro_rules! impl_tiff_sample {
    ($ty:ty, $variant:ident) => {
        impl TiffSample for $ty {
            const SAMPLE_TYPE: DecodingSampleType = DecodingSampleType::$variant;
            fn as_buffer(buf: &mut [Self]) -> DecodingBuffer<'_> {
                DecodingBuffer::$variant(buf)
            }
        }
    };
}

impl_tiff_sample!(u8, U8);
impl_tiff_sample!(u16, U16);
impl_tiff_sample!(f32, F32);

// Decodes a TIFF into `dst`, checking its size, channel count and sample type first.
// The typed samples are exposed to tiff as native-endian bytes (exactly what its own
// `read_image` does), so no temporary byte buffer or per-sample conversion is needed.
fn decode_tiff_impl<T: TiffSample, const C: usize>(
    src: &[u8],
    dst: &mut Image<T, C>,
) -> Result<(), IoError> {
    let mut decoder = tiff::decoder::Decoder::new(Cursor::new(src))?;

    let (width, height) = decoder.dimensions()?;
    if width as usize != dst.width() || height as usize != dst.height() {
        return Err(IoError::DecodeMismatchResolution(
            height as usize,
            width as usize,
            dst.height(),
            dst.width(),
        ));
    }

    check_tiff_format(&mut decoder, C as u8, T::SAMPLE_TYPE)?;

    let expected_len = dst.size().checked_len(C)?;
    let dst = dst.as_slice_mut();
    if dst.len() != expected_len {
        return Err(IoError::InvalidBufferSize(dst.len(), expected_len));
    }

    decoder.read_image_bytes(T::as_buffer(dst).as_bytes_mut())?;
    Ok(())
}

/// Write a TIFF image with a RGB8 color type.
///
/// # Arguments
///
/// * `file_path` - The path to the TIFF image.
/// * `image` - The Rgb8 image to write.
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
/// * `image` - The Gray8 image to write.
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
/// * `image` - The Rgb16 image to write.
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
/// * `image` - The Gray16 image to write.
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
/// * `image` - The Grayf32 image to write.
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
/// * `image` - The Rgbf32 image to write.
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
    write_tiff_into::<_, C, T>(file, image_data, image_size)
}

/// Encodes a TIFF image into any `Write + Seek` target. Drives both the
/// file-path writers and the buffer-based `encode_image_tiff_*` API.
fn write_tiff_into<W, C, T>(
    writer: W,
    image_data: &[T],
    image_size: ImageSize,
) -> Result<(), IoError>
where
    W: std::io::Write + std::io::Seek,
    C: tiff::encoder::colortype::ColorType<Inner = T>,
    [T]: tiff::encoder::TiffValue,
{
    let mut encoder = TiffEncoder::new(writer)?;
    encoder.write_image::<C>(
        image_size.width as u32,
        image_size.height as u32,
        image_data,
    )?;
    Ok(())
}

/// Reserve approximate uncompressed-TIFF capacity in `buffer` (raw bytes plus
/// ~1 KB header) so the inner `Cursor<&mut Vec<u8>>` doesn't trigger repeated
/// `Vec` doublings during `TiffEncoder::write_image`.
fn reserve_tiff<T: Sized>(buffer: &mut Vec<u8>, slice_len: usize) {
    buffer.reserve(slice_len * std::mem::size_of::<T>() + 1024);
}

/// Encodes an RGB u8 image as TIFF bytes into `buffer` (appended).
pub fn encode_image_tiff_rgb8(image: &Image<u8, 3>, buffer: &mut Vec<u8>) -> Result<(), IoError> {
    reserve_tiff::<u8>(buffer, image.as_slice().len());
    let mut cursor = std::io::Cursor::new(buffer);
    write_tiff_into::<_, colortype::RGB8, u8>(&mut cursor, image.as_slice(), image.size())
}

/// Encodes a grayscale u8 image as TIFF bytes into `buffer` (appended).
pub fn encode_image_tiff_mono8(image: &Image<u8, 1>, buffer: &mut Vec<u8>) -> Result<(), IoError> {
    reserve_tiff::<u8>(buffer, image.as_slice().len());
    let mut cursor = std::io::Cursor::new(buffer);
    write_tiff_into::<_, colortype::Gray8, u8>(&mut cursor, image.as_slice(), image.size())
}

/// Encodes an RGB u16 image as TIFF bytes into `buffer` (appended).
pub fn encode_image_tiff_rgb16(image: &Image<u16, 3>, buffer: &mut Vec<u8>) -> Result<(), IoError> {
    reserve_tiff::<u16>(buffer, image.as_slice().len());
    let mut cursor = std::io::Cursor::new(buffer);
    write_tiff_into::<_, colortype::RGB16, u16>(&mut cursor, image.as_slice(), image.size())
}

/// Encodes a grayscale u16 image as TIFF bytes into `buffer` (appended).
pub fn encode_image_tiff_mono16(
    image: &Image<u16, 1>,
    buffer: &mut Vec<u8>,
) -> Result<(), IoError> {
    reserve_tiff::<u16>(buffer, image.as_slice().len());
    let mut cursor = std::io::Cursor::new(buffer);
    write_tiff_into::<_, colortype::Gray16, u16>(&mut cursor, image.as_slice(), image.size())
}

/// Encodes a grayscale f32 image as TIFF bytes into `buffer` (appended).
pub fn encode_image_tiff_mono32f(
    image: &Image<f32, 1>,
    buffer: &mut Vec<u8>,
) -> Result<(), IoError> {
    reserve_tiff::<f32>(buffer, image.as_slice().len());
    let mut cursor = std::io::Cursor::new(buffer);
    write_tiff_into::<_, colortype::Gray32Float, f32>(&mut cursor, image.as_slice(), image.size())
}

/// Encodes an RGB f32 image as TIFF bytes into `buffer` (appended).
pub fn encode_image_tiff_rgb32f(
    image: &Image<f32, 3>,
    buffer: &mut Vec<u8>,
) -> Result<(), IoError> {
    reserve_tiff::<f32>(buffer, image.as_slice().len());
    let mut cursor = std::io::Cursor::new(buffer);
    write_tiff_into::<_, colortype::RGB32Float, f32>(&mut cursor, image.as_slice(), image.size())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::error::IoError;
    use std::fs::{create_dir_all, read};

    #[test]
    fn decode_rejects_mismatched_pixel_format() -> Result<(), Box<dyn std::error::Error>> {
        let size = ImageSize {
            width: 4,
            height: 4,
        };
        let gray = Gray8::from_size_val(size, 200)?;
        let mut encoded = Vec::new();
        encode_image_tiff_mono8(&gray, &mut encoded)?;

        let mut rgb = Rgb8::from_size_val(size, 0)?;
        assert!(matches!(
            decode_image_tiff_rgb8(&encoded, &mut rgb),
            Err(IoError::FormatMismatch(_))
        ));
        let mut mono16 = Gray16::from_size_val(size, 0)?;
        assert!(matches!(
            decode_image_tiff_mono16(&encoded, &mut mono16),
            Err(IoError::FormatMismatch(_))
        ));
        let mut back = Gray8::from_size_val(size, 0)?;
        decode_image_tiff_mono8(&encoded, &mut back)?;
        assert_eq!(back.as_slice(), gray.as_slice());
        Ok(())
    }

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
        )?;

        let file_path = tmp_dir.path().join("rgb32f.tiff");
        write_image_tiff_rgb32f(&file_path, &img_rgb32f)?;

        let img_rgb32f_back = read_image_tiff_rgb32f(&file_path)?;
        assert_eq!(img_rgb32f_back.as_slice(), img_rgb32f.as_slice());

        Ok(())
    }

    #[test]
    fn decode_tiff_rgb8() -> Result<(), IoError> {
        let bytes = read("../../tests/data/dog.tiff")?;
        let layout = decode_image_tiff_layout(&bytes)?;
        assert_eq!(layout.image_size.width, 258);
        assert_eq!(layout.image_size.height, 195);
        assert_eq!(layout.channels, 3);

        let mut image = Rgb8::from_size_val(layout.image_size, 0)?;
        decode_image_tiff_rgb8(&bytes, &mut image)?;

        assert_eq!(image.cols(), layout.image_size.width);
        assert_eq!(image.rows(), layout.image_size.height);
        assert_eq!(image.num_channels(), layout.channels as usize);

        Ok(())
    }

    #[test]
    fn decode_tiff_rgb16() -> Result<(), IoError> {
        let bytes = read("../../tests/data/rgb16.tiff")?;
        let layout = decode_image_tiff_layout(&bytes)?;
        assert_eq!(layout.image_size.width, 32);
        assert_eq!(layout.image_size.height, 32);
        assert_eq!(layout.channels, 3);

        let mut image = Rgb16::from_size_val(layout.image_size, 0)?;
        decode_image_tiff_rgb16(&bytes, &mut image)?;

        assert_eq!(image.cols(), layout.image_size.width);
        assert_eq!(image.rows(), layout.image_size.height);
        assert_eq!(image.num_channels(), layout.channels as usize);

        Ok(())
    }

    #[test]
    fn decode_tiff_rgb32f() -> Result<(), IoError> {
        let bytes = read("../../tests/data/rgb32.tiff")?;
        let layout = decode_image_tiff_layout(&bytes)?;
        assert_eq!(layout.image_size.width, 32);
        assert_eq!(layout.image_size.height, 32);
        assert_eq!(layout.channels, 3);

        let mut image = Rgbf32::from_size_val(layout.image_size, 0.0f32)?;
        decode_image_tiff_rgb32f(&bytes, &mut image)?;

        assert_eq!(image.cols(), layout.image_size.width);
        assert_eq!(image.rows(), layout.image_size.height);
        assert_eq!(image.num_channels(), layout.channels as usize);

        Ok(())
    }
}
