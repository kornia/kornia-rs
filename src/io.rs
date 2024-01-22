use std::path::Path;

use pyo3::prelude::*;

use image;
use sophus_rs::image::mut_image::{MutImage2F32, MutImage3F32, MutImage3U8};
use turbojpeg;
use sophus_rs::image::view::ImageSize as _ImageSize;
use sophus_rs::tensor::mut_tensor::MutTensor;

// internal libs
use crate::tensor::Tensor;

//#[pyclass]
//pub struct ImageSize {
//    #[pyo3(get)]
//    pub width: usize,
//    #[pyo3(get)]
//    pub height: usize,
//}

// wrap the ImageSize struct from sophus_rs
#[pyclass]
pub struct ImageSize {
    inner: _ImageSize,
}

#[pymethods]
impl ImageSize {
    #[new]
    pub fn new(width: usize, height: usize) -> Self {
        ImageSize {
            inner: _ImageSize{ width, height },
        }
    }

    #[getter]
    pub fn width(&self) -> usize {
        self.inner.width
    }

    #[getter]
    pub fn height(&self) -> usize {
        self.inner.height
    }
}

#[pyclass]
pub struct ImageDecoder {
    pub decompressor: turbojpeg::Decompressor,
}

#[pyclass]
pub struct ImageEncoder {
    pub compressor: turbojpeg::Compressor,
}

#[pymethods]
impl ImageEncoder {
    #[new]
    pub fn new() -> ImageEncoder {
        let compressor = turbojpeg::Compressor::new().unwrap();
        ImageEncoder { compressor }
    }

    pub fn encode(&mut self, data: &[u8], shape: [usize; 3]) -> Vec<u8> {
        let image = turbojpeg::Image {
            pixels: data,
            width: shape[1],
            pitch: 3 * shape[1],
            height: shape[0],
            format: turbojpeg::PixelFormat::RGB,
        };

        let jpeg_data = self.compressor.compress_to_vec(image);

        jpeg_data.unwrap()
    }

    pub fn set_quality(&mut self, quality: i32) {
        self.compressor.set_quality(quality)
    }
}

impl Default for ImageEncoder {
    fn default() -> Self {
        Self::new()
    }
}

#[pymethods]
impl ImageDecoder {
    #[new]
    pub fn new() -> ImageDecoder {
        let decompressor = turbojpeg::Decompressor::new().unwrap();
        ImageDecoder { decompressor }
    }

    pub fn read_header(&mut self, jpeg_data: &[u8]) -> ImageSize {
        // read the JPEG header with image size
        let header = self.decompressor.read_header(jpeg_data).unwrap();
        ImageSize::new(header.width, header.height)
    }

    pub fn decode(&mut self, jpeg_data: &[u8]) -> Tensor {
        // get the image size to allocate th data storage
        let image_size: ImageSize = self.read_header(jpeg_data);

        // prepare a storage for the raw pixel data
        let mut pixels = vec![0; image_size.height() * image_size.width() * 3];

        // allocate image container
        let image = turbojpeg::Image {
            pixels: pixels.as_mut_slice(),
            width: image_size.width(),
            pitch: 3 * image_size.width(), // we use no padding between rows
            height: image_size.height(),
            format: turbojpeg::PixelFormat::RGB,
        };

        // decompress the JPEG data
        self.decompressor.decompress(jpeg_data, image).unwrap();

        // return the raw pixel data and shape as Tensor
        let shape = vec![image_size.height() as i64, image_size.width() as i64, 3];

        Tensor::new(shape, pixels)
    }
}

impl Default for ImageDecoder {
    fn default() -> Self {
        Self::new()
    }
}

#[pyfunction]
pub fn read_image_jpeg(file_path: String) -> Tensor {
    // get the JPEG data
    let jpeg_data = std::fs::read(file_path).unwrap();

    // create the decoder
    let mut decoder = ImageDecoder::new();

    // decode the data directly from a file
    decoder.decode(&jpeg_data)
}

#[pyfunction]
pub fn write_image_jpeg(file_path: String, jpeg_data: Vec<u8>) {
    let _bytes_written = std::fs::write(file_path, jpeg_data);
}

#[pyfunction]
pub fn read_image_rs(file_path: String) -> Tensor {
    let img: image::DynamicImage = image::open(file_path).unwrap();
    let data = img.to_rgb8().to_vec();
    let shape = vec![img.height() as i64, img.width() as i64, 3];
    Tensor::new(shape, data)
}

pub fn read_image(file_path: &Path) -> MutImage3U8 {
    let img: image::DynamicImage = image::open(file_path.to_str().unwrap()).unwrap();
    let data = img.as_bytes();
    MutImage3U8::from_image_size_and_val(_ImageSize { 
        width: img.width() as usize, 
        height: img.height() as usize 
    }, data)
}

#[cfg(test)]
mod tests {
    #[test]
    fn image_size() {
        let image_size = super::ImageSize::new(10, 20);
        assert_eq!(image_size.width(), 10);
        assert_eq!(image_size.height(), 20);
    }
}
