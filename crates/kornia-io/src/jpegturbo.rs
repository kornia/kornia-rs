use std::sync::Mutex;

use kornia_image::{Image, ImageError, ImageSize};

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
}

/// A JPEG decoder using the turbojpeg library.
pub struct JpegTurboDecoder(Mutex<turbojpeg::Decompressor>);

/// A JPEG encoder using the turbojpeg library.
pub struct JpegTurboEncoder(Mutex<turbojpeg::Compressor>);

impl Default for JpegTurboDecoder {
    fn default() -> Self {
        match Self::new() {
            Ok(decoder) => decoder,
            Err(e) => panic!("Failed to create ImageDecoder: {}", e),
        }
    }
}

impl Default for JpegTurboEncoder {
    fn default() -> Self {
        match Self::new() {
            Ok(encoder) => encoder,
            Err(e) => panic!("Failed to create ImageEncoder: {}", e),
        }
    }
}

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
    pub fn encode_rgb8(&self, image: &Image<u8, 3>) -> Result<Vec<u8>, JpegTurboError> {
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
            .expect("Failed to lock the compressor")
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
            .expect("Failed to lock the compressor")
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
            .expect("Failed to lock the decompressor")
            .read_header(jpeg_data)?;

        Ok(ImageSize {
            width: header.width,
            height: header.height,
        })
    }

    /// Decodes the given JPEG data as RGB8 image.
    ///
    /// # Arguments
    ///
    /// * `jpeg_data` - The JPEG data to decode.
    ///
    /// # Returns
    ///
    /// The decoded data as Image<u8, 3>.
    pub fn decode_rgb8(&self, jpeg_data: &[u8]) -> Result<Image<u8, 3>, JpegTurboError> {
        // get the image size to allocate th data storage
        let image_size = self.read_header(jpeg_data)?;

        // prepare a storage for the raw pixel data
        let mut pixels = vec![0u8; image_size.height * image_size.width * 3];

        // allocate image container
        let buf = turbojpeg::Image {
            pixels: pixels.as_mut_slice(),
            width: image_size.width,
            pitch: 3 * image_size.width, // we use no padding between rows
            height: image_size.height,
            format: turbojpeg::PixelFormat::RGB,
        };

        // decompress the JPEG data
        self.0
            .lock()
            .expect("Failed to lock the decompressor")
            .decompress(jpeg_data, buf)?;

        Ok(Image::new(image_size, pixels)?)
    }
}

#[cfg(test)]
mod tests {
    use crate::jpegturbo::{JpegTurboDecoder, JpegTurboEncoder, JpegTurboError};

    #[test]
    fn image_decoder() -> Result<(), JpegTurboError> {
        let jpeg_data = std::fs::read("../../tests/data/dog.jpeg").unwrap();
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
}
