use pyo3::prelude::*;

// for libjpeg-turbo
use image;
use turbojpeg::{Decompressor, Image, PixelFormat};

// internal libs
use crate::tensor::cv;

// implementation function for libjpeg-turbo to load images
fn _read_image_jpeg_impl(
    file_path: String,
) -> Result<(Vec<u8>, Vec<i64>), Box<dyn std::error::Error>> {
    // get the JPEG data
    let jpeg_data = std::fs::read(file_path)?;

    // initialize a Decompressor
    let mut decompressor = Decompressor::new()?;

    // read the JPEG header with image size
    let header = decompressor.read_header(&jpeg_data)?;
    let (width, height) = (header.width, header.height);

    // prepare a storage for the raw pixel data
    let mut pixels = vec![0; height * width * 3];
    let image = Image {
        pixels: pixels.as_mut_slice(),
        width,
        pitch: 3 * width, // we use no padding between rows
        height,
        format: PixelFormat::RGB,
    };

    // decompress the JPEG data
    decompressor.decompress_to_slice(&jpeg_data, image)?;

    // return the raw pixel data and shape
    Ok((pixels, vec![height as i64, width as i64, 3]))
}

#[pyfunction]
pub fn read_image_jpeg(file_path: String) -> cv::Tensor {
    // decode image and return tuple with data and shape
    let (data, shape) = _read_image_jpeg_impl(file_path).unwrap();
    cv::Tensor::new(shape, data)
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
