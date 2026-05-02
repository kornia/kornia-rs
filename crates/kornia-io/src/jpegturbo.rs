use crate::error::IoError;
use kornia_image::{
    allocator::{CpuAllocator, ImageAllocator},
    Image, ImageError, ImageSize,
};
use std::{path::Path, sync::Mutex};
use turbojpeg;

/// Error types for the JPEG module.
#[derive(thiserror::Error, Debug)]
pub enum JpegTurboError {
    /// Error when the JPEG compressor cannot be created.
    #[error("Something went wrong with the JPEG compressor")]
    TurboJpegError(#[from] turbojpeg::Error),

    /// Error when the image data is not contiguous.
    #[error("Image data is not contiguous")]
    ImageDataNotContiguous,

    /// Error to create the image.
    #[error("Failed to create image")]
    ImageCreationError(#[from] ImageError),

    /// Error when mutex is poisoned.
    #[error("Mutex is poisoned")]
    MutexPoisoned,

    /// I/O error, e.g. when reading a JPEG file from disk.
    #[error(transparent)]
    IoError(#[from] std::io::Error),
}

/// A JPEG decoder using the turbojpeg library.
/// **Note on migration:** The `Default` trait implementation was removed. Please use [`JpegTurboDecoder::new()`] instead.
pub struct JpegTurboDecoder(Mutex<turbojpeg::Decompressor>);

/// A JPEG encoder using the turbojpeg library.
/// **Note on migration:** The `Default` trait implementation was removed. Please use [`JpegTurboEncoder::new()`] instead.
pub struct JpegTurboEncoder(Mutex<turbojpeg::Compressor>);

// Implementations for ImageDecoder and ImageEncoder

/// Implementation of the ImageEncoder struct.
impl JpegTurboEncoder {
    /// Creates a new `ImageEncoder`.
    ///
    /// # Returns
    ///
    /// A new `ImageEncoder` instance.
    ///
    /// # Panics
    ///
    /// Panics if the compressor cannot be created.
    pub fn new() -> Result<Self, JpegTurboError> {
        let compressor = turbojpeg::Compressor::new()?;
        Ok(Self(Mutex::new(compressor)))
    }

    /// Encodes the given RGB8 image into a JPEG image.
    ///
    /// # Arguments
    ///
    /// * `image` - The image to encode.
    ///
    /// # Returns
    ///
    /// The encoded data as `Vec<u8>`.
    pub fn encode_rgb8<A: ImageAllocator>(
        &self,
        image: &Image<u8, 3, A>,
    ) -> Result<Vec<u8>, JpegTurboError> {
        // get the image data
        let image_data = image.as_slice();

        // create a turbojpeg image
        let buf = turbojpeg::Image {
            pixels: image_data,
            width: image.width(),
            pitch: 3 * image.width(),
            height: image.height(),
            format: turbojpeg::PixelFormat::RGB,
        };

        // encode the image
        Ok(self
            .0
            .lock()
            .map_err(|_| JpegTurboError::MutexPoisoned)?
            .compress_to_vec(buf)?)
    }

    /// Sets the quality of the encoder.
    ///
    /// # Arguments
    ///
    /// * `quality` - The quality to set.
    pub fn set_quality(&self, quality: i32) -> Result<(), JpegTurboError> {
        Ok(self
            .0
            .lock()
            .map_err(|_| JpegTurboError::MutexPoisoned)?
            .set_quality(quality)?)
    }
}

/// Implementation of the ImageDecoder struct.
impl JpegTurboDecoder {
    /// Creates a new `ImageDecoder`.
    ///
    /// # Returns
    ///
    /// A new `ImageDecoder` instance.
    pub fn new() -> Result<Self, JpegTurboError> {
        let decompressor = turbojpeg::Decompressor::new()?;
        Ok(JpegTurboDecoder(Mutex::new(decompressor)))
    }

    /// Reads the header of a JPEG image.
    ///
    /// # Arguments
    ///
    /// * `jpeg_data` - The JPEG data to read the header from.
    ///
    /// # Returns
    ///
    /// The image size.
    ///
    /// # Panics
    ///
    /// Panics if the header cannot be read.
    pub fn read_header(&self, jpeg_data: &[u8]) -> Result<ImageSize, JpegTurboError> {
        // read the JPEG header with image size
        let header = self
            .0
            .lock()
            .map_err(|_| JpegTurboError::MutexPoisoned)?
            .read_header(jpeg_data)?;

        Ok(ImageSize {
            width: header.width,
            height: header.height,
        })
    }

    /// Decodes the given JPEG data as RGB8 image.
    pub fn decode_rgb8(
        &self,
        jpeg_data: &[u8],
    ) -> Result<Image<u8, 3, CpuAllocator>, JpegTurboError> {
        let image_size = self.read_header(jpeg_data)?;
        let mut dst = Image::from_size_val(image_size, 0u8, CpuAllocator)?;
        self.decode_rgb8_into(jpeg_data, &mut dst)?;
        Ok(dst)
    }

    /// Decodes the given JPEG data as Gray/Mono8 image.
    pub fn decode_gray8(
        &self,
        jpeg_data: &[u8],
    ) -> Result<Image<u8, 1, CpuAllocator>, JpegTurboError> {
        let image_size = self.read_header(jpeg_data)?;
        let mut dst = Image::from_size_val(image_size, 0u8, CpuAllocator)?;
        self.decode_gray8_into(jpeg_data, &mut dst)?;
        Ok(dst)
    }

    /// Decodes JPEG bytes as RGB8 into a pre-allocated buffer.
    pub fn decode_rgb8_into<A: ImageAllocator>(
        &self,
        jpeg_data: &[u8],
        dst: &mut Image<u8, 3, A>,
    ) -> Result<(), JpegTurboError> {
        let size = dst.size();
        self.decode_into(
            jpeg_data,
            dst.as_slice_mut(),
            size,
            turbojpeg::PixelFormat::RGB,
        )
    }

    /// Decodes JPEG bytes as Gray/Mono8 into a pre-allocated buffer.
    pub fn decode_gray8_into<A: ImageAllocator>(
        &self,
        jpeg_data: &[u8],
        dst: &mut Image<u8, 1, A>,
    ) -> Result<(), JpegTurboError> {
        let size = dst.size();
        self.decode_into(
            jpeg_data,
            dst.as_slice_mut(),
            size,
            turbojpeg::PixelFormat::GRAY,
        )
    }

    fn decode_into(
        &self,
        jpeg_data: &[u8],
        pixels: &mut [u8],
        image_size: ImageSize,
        format: turbojpeg::PixelFormat,
    ) -> Result<(), JpegTurboError> {
        let header_size = self.read_header(jpeg_data)?;
        if header_size != image_size {
            return Err(JpegTurboError::ImageCreationError(
                ImageError::InvalidImageSize(
                    header_size.width,
                    header_size.height,
                    image_size.width,
                    image_size.height,
                ),
            ));
        }

        let pitch = format.size() * image_size.width;
        let buf = turbojpeg::Image {
            pixels,
            width: image_size.width,
            pitch,
            height: image_size.height,
            format,
        };

        self.0
            .lock()
            .map_err(|_| JpegTurboError::MutexPoisoned)?
            .decompress(jpeg_data, buf)?;
        Ok(())
    }
}

/// Reads a JPEG image in `RGB8` format from the given file path.
///
/// The method reads the JPEG image data directly from a file leveraging the libjpeg-turbo library.
///
/// # Arguments
///
/// * `image_path` - The path to the JPEG image.
///
/// # Returns
///
/// An in image containing the JPEG image data.
///
/// # Example
///
/// ```
/// use kornia_image::Image;
/// use kornia_io::jpegturbo as F;
///
/// let image: Image<u8, 3, _> = F::read_image_jpegturbo_rgb8("../../tests/data/dog.jpeg").unwrap();
///
/// assert_eq!(image.cols(), 258);
/// assert_eq!(image.rows(), 195);
/// assert_eq!(image.num_channels(), 3);
/// ```
pub fn read_image_jpegturbo_rgb8(
    file_path: impl AsRef<Path>,
) -> Result<Image<u8, 3, CpuAllocator>, IoError> {
    let file_path = file_path.as_ref().to_owned();
    // verify the file exists and is a JPEG
    if !file_path.exists() {
        return Err(IoError::FileDoesNotExist(file_path.to_path_buf()));
    }

    if file_path
        .extension()
        .is_none_or(|ext| !ext.eq_ignore_ascii_case("jpg") && !ext.eq_ignore_ascii_case("jpeg"))
    {
        return Err(IoError::InvalidFileExtension(file_path.to_path_buf()));
    }

    // open the file and map it to memory
    let jpeg_data = std::fs::read(file_path)?;

    // decode the data directly from memory
    let image = {
        let decoder = JpegTurboDecoder::new()?;
        decoder.decode_rgb8(&jpeg_data)?
    };

    Ok(image)
}

/// Writes the given JPEG data to the given file path.
///
/// # Arguments
///
/// * `file_path` - The path to the JPEG image.
/// * `image` - The tensor containing the JPEG image data.
/// * `quality` - The quality of the JPEG encoding, range from 0 (lowest) to 100 (highest)
pub fn write_image_jpegturbo_rgb8<A: ImageAllocator>(
    file_path: impl AsRef<Path>,
    image: &Image<u8, 3, A>,
    quality: u8,
) -> Result<(), IoError> {
    let file_path = file_path.as_ref().to_owned();

    // compress the image
    let encoder = JpegTurboEncoder::new()?;
    encoder.set_quality(quality as i32)?;

    let jpeg_data = encoder.encode_rgb8(image)?;

    // write the data directly to a file
    std::fs::write(file_path, jpeg_data)?;

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::error::IoError;

    #[test]
    fn image_decoder() -> Result<(), JpegTurboError> {
        let jpeg_data = std::fs::read("../../tests/data/dog.jpeg")?;
        // read the header
        let image_size = JpegTurboDecoder::new()?.read_header(&jpeg_data)?;
        assert_eq!(image_size.width, 258);
        assert_eq!(image_size.height, 195);
        // load the image as file and decode it
        let image = JpegTurboDecoder::new()?.decode_rgb8(&jpeg_data)?;
        assert_eq!(image.cols(), 258);
        assert_eq!(image.rows(), 195);
        assert_eq!(image.num_channels(), 3);
        Ok(())
    }

    #[test]
    fn image_decoder_gray8() -> Result<(), JpegTurboError> {
        let jpeg_data = std::fs::read("../../tests/data/dog.jpeg")?;
        let image = JpegTurboDecoder::new()?.decode_gray8(&jpeg_data)?;
        assert_eq!(image.cols(), 258);
        assert_eq!(image.rows(), 195);
        assert_eq!(image.num_channels(), 1);
        assert_eq!(image.as_slice().len(), 258 * 195);
        Ok(())
    }

    #[test]
    fn image_encoder() -> Result<(), Box<dyn std::error::Error>> {
        let jpeg_data_fs = std::fs::read("../../tests/data/dog.jpeg")?;
        let image = JpegTurboDecoder::new()?.decode_rgb8(&jpeg_data_fs)?;
        let jpeg_data = JpegTurboEncoder::new()?.encode_rgb8(&image)?;
        let image_back = JpegTurboDecoder::new()?.decode_rgb8(&jpeg_data)?;
        assert_eq!(image_back.cols(), 258);
        assert_eq!(image_back.rows(), 195);
        assert_eq!(image_back.num_channels(), 3);
        Ok(())
    }

    #[test]
    fn read_jpeg() -> Result<(), IoError> {
        let image = read_image_jpegturbo_rgb8("../../tests/data/dog.jpeg")?;
        assert_eq!(image.cols(), 258);
        assert_eq!(image.rows(), 195);
        Ok(())
    }

    #[test]
    fn read_write_jpeg() -> Result<(), IoError> {
        let tmp_dir = tempfile::tempdir()?;
        std::fs::create_dir_all(tmp_dir.path())?;

        let file_path = tmp_dir.path().join("dog.jpeg");
        let image_data = read_image_jpegturbo_rgb8("../../tests/data/dog.jpeg")?;
        write_image_jpegturbo_rgb8(&file_path, &image_data, 100)?;

        let image_data_back = read_image_jpegturbo_rgb8(&file_path)?;
        assert!(file_path.exists(), "File does not exist: {file_path:?}");

        assert_eq!(image_data_back.cols(), 258);
        assert_eq!(image_data_back.rows(), 195);
        assert_eq!(image_data_back.num_channels(), 3);

        Ok(())
    }
}
