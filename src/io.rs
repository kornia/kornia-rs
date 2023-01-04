use pyo3::prelude::*;

use image;
use turbojpeg;

// internal libs
use crate::tensor::cv;

#[pyclass]
pub struct ImageSize {
    #[pyo3(get)]
    pub width: usize,
    #[pyo3(get)]
    pub height: usize,
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
        ImageSize {
            width: header.width,
            height: header.height,
        }
    }

    pub fn decode(&mut self, jpeg_data: &[u8]) -> cv::Tensor {
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
        self.decompressor.decompress(jpeg_data, image).unwrap();

        // return the raw pixel data and shape as Tensor
        let shape = vec![image_size.height as i64, image_size.width as i64, 3];

        cv::Tensor::new(shape, pixels)
    }
}

impl Default for ImageDecoder {
    fn default() -> Self {
        Self::new()
    }
}

#[pyfunction]
pub fn read_image_jpeg(file_path: String) -> cv::Tensor {
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
pub fn read_image_rs(file_path: String) -> cv::Tensor {
    let img: image::DynamicImage = image::open(file_path).unwrap();
    let data = img.to_rgb8().to_vec();
    let shape = vec![img.height() as i64, img.width() as i64, 3];
    cv::Tensor::new(shape, data)
}

//#[cfg(test)]
//mod tests {
//    use super::*;
//    use std::path::PathBuf;
//    use std::time::SystemTime;
//    use test::Bencher;
//
//    #[test]
//    fn load() {
//        let path: PathBuf = [env!("CARGO_MANIFEST_DIR"), "clients", "test.jpg"]
//            .iter()
//            .collect();
//
//        let str_path = path.into_os_string().into_string().unwrap();
//        let start = SystemTime::now()
//            .duration_since(SystemTime::UNIX_EPOCH)
//            .expect("get millis error");
//        let info = read_image_rs(str_path.clone());
//        let end = SystemTime::now()
//            .duration_since(SystemTime::UNIX_EPOCH)
//            .expect("get millis error");
//        println!("{}", str_path);
//        println!("{:?}", info);
//        println!(
//            "time {:?} secs",
//            (end.as_millis() - start.as_millis()) as f64 / 1000.,
//        );
//    }
//
//    #[bench]
//    fn bench(b: &mut Bencher) {
//        let path: PathBuf = [env!("CARGO_MANIFEST_DIR"), "clients", "test.jpg"]
//            .iter()
//            .collect();
//        let str_path = path.into_os_string().into_string().unwrap();
//        b.iter(|| {
//            let info = read_image_rs(str_path.clone());
//        });
//    }
//}
