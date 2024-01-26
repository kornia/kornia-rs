use turbojpeg;

use crate::image::{ImageSize, Image};

/// A JPEG decoder using the turbojpeg library.
pub struct ImageDecoder {
    pub decompressor: turbojpeg::Decompressor,
}

/// A JPEG encoder using the turbojpeg library.
pub struct ImageEncoder {
    pub compressor: turbojpeg::Compressor,
}

impl Default for ImageDecoder {
    fn default() -> Self {
        Self::new()
    }
}

impl Default for ImageEncoder {
    fn default() -> Self {
        Self::new()
    }
}

/// Implementation of the ImageEncoder struct.
impl ImageEncoder {
    /// Creates a new `ImageEncoder`.
    ///
    /// # Returns
    ///
    /// A new `ImageEncoder` instance.
    ///
    /// # Panics
    ///
    /// Panics if the compressor cannot be created.
    pub fn new() -> ImageEncoder {
        let compressor = match turbojpeg::Compressor::new() {
            Ok(c) => c,
            Err(e) => panic!("Error creating compressor: {}", e),
        };
        ImageEncoder { compressor }
    }

    /// Encodes the given data into a JPEG image.
    ///
    /// # Arguments
    ///
    /// * `image` - The image to encode.
    /// 
    /// # Returns
    /// 
    /// The encoded data as Vec<u8>.
    pub fn encode(&mut self, image: &Image) -> Vec<u8> {
        // get the image data
        let image_data = match image.data.as_slice() {
            Some(d) => d,
            None => panic!("Image data is not contiguous"),
        };

        // create a turbojpeg image
        let _image = turbojpeg::Image {
            pixels: image_data,
            width: image.image_size().width,
            pitch: 3 * image.image_size().width,
            height: image.image_size().height,
            format: turbojpeg::PixelFormat::RGB,
        };

        // encode the image
        match self.compressor.compress_to_vec(_image) {
            Ok(d) => d,
            Err(e) => panic!("Error compressing image: {}", e),
        }
    }

    /// Sets the quality of the encoder.
    ///
    /// # Arguments
    ///
    /// * `quality` - The quality to set.
    pub fn set_quality(&mut self, quality: i32) {
        self.compressor.set_quality(quality)
    }
}

/// Implementation of the ImageDecoder struct.
impl ImageDecoder {
    /// Creates a new `ImageDecoder`.
    ///
    /// # Returns
    ///
    /// A new `ImageDecoder` instance.
    pub fn new() -> ImageDecoder {
        let decompressor = match turbojpeg::Decompressor::new() {
            Ok(d) => d,
            Err(e) => panic!("Error creating decompressor: {}", e),
        };
        ImageDecoder { decompressor }
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
    pub fn read_header(&mut self, jpeg_data: &[u8]) -> ImageSize {
        // read the JPEG header with image size
        let header = match self.decompressor.read_header(jpeg_data) {
            Ok(h) => h,
            Err(e) => panic!("Error reading header: {}", e),
        };
        ImageSize {
            width: header.width,
            height: header.height,
        }
    }

    /// Decodes the given JPEG data.
    ///
    /// # Arguments
    ///
    /// * `jpeg_data` - The JPEG data to decode.
    ///
    /// # Returns
    ///
    /// The decoded data as Tensor.
    pub fn decode(&mut self, jpeg_data: &[u8]) -> Image {
        // get the image size to allocate th data storage
        let image_size: ImageSize = self.read_header(jpeg_data);

        // prepare a storage for the raw pixel data
        let mut pixels = vec![0; image_size.height * image_size.width * 3];

        // allocate image container
        let image = turbojpeg::Image {
            pixels: pixels.as_mut_slice(),
            width: image_size.width,
            pitch: 3 * image_size.width, // we use no padding between rows
            height: image_size.height,
            format: turbojpeg::PixelFormat::RGB,
        };

        // decompress the JPEG data
        match self.decompressor.decompress(jpeg_data, image) {
            Ok(_) => {}
            Err(e) => panic!("Error decompressing image: {}", e),
        };

        Image::new(image_size, pixels)
    }
}

#[cfg(test)]
mod tests {
    use crate::io::jpeg::{ImageDecoder, ImageEncoder};

    #[test]
    fn image_decoder() {
        let mut decoder = ImageDecoder::new();
        let jpeg_data = std::fs::read("tests/data/dog.jpeg").unwrap();
        // read the header
        let image_size = decoder.read_header(&jpeg_data);
        assert_eq!(image_size.width, 258);
        assert_eq!(image_size.height, 195);
        // load the image as file and decode it
        let image = decoder.decode(&jpeg_data);
        assert_eq!(image.image_size().width, 258);
        assert_eq!(image.image_size().height, 195);
        assert_eq!(image.num_channels(), 3);
    }

    #[test]
    fn image_encoder() {
        let mut decoder = ImageDecoder::new();
        let jpeg_data_fs = std::fs::read("tests/data/dog.jpeg").unwrap();
        let image = decoder.decode(&jpeg_data_fs);
        let mut encoder = ImageEncoder::new();
        //let data_slice: &[u8] = &image.data.as_slice().unwrap();
        let jpeg_data = encoder.encode(&image);
        let image_back = decoder.decode(&jpeg_data);
        assert_eq!(image_back.image_size().width, 258);
        assert_eq!(image_back.image_size().height, 195);
        assert_eq!(image_back.num_channels(), 3);
    }
}
