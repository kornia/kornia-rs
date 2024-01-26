use std::path::Path;

use crate::image::{Image, ImageSize};

use super::jpeg::{ImageDecoder, ImageEncoder};

/// Reads a JPEG image from the given file path.
///
/// # Arguments
///
/// * `image_path` - The path to the JPEG image.
///
/// # Returns
///
/// A tensor containing the JPEG image data.
pub fn read_image_jpeg(file_path: &Path) -> Image {
    // verify the file exists and is a JPEG
    if !file_path.exists() {
        panic!("File does not exist: {}", file_path.to_str().unwrap());
    }

    let file_path = match file_path.extension() {
        Some(ext) => {
            if ext == "jpg" || ext == "jpeg" {
                file_path
            } else {
                panic!("File is not a JPEG: {}", file_path.to_str().unwrap());
            }
        }
        None => {
            panic!("File has no extension: {}", file_path.to_str().unwrap());
        }
    };

    // decode the data directly from a file

    match std::fs::read(file_path) {
        Ok(data) => {
            let mut decoder = ImageDecoder::new();
            decoder.decode(&data)
        }
        Err(e) => panic!("Error reading file: {}", e),
    }
}

/// Writes the given JPEG data to the given file path.
///
/// # Arguments
///
/// * `file_path` - The path to the JPEG image.
/// * `image` - The tensor containing the JPEG image data.
pub fn write_image_jpeg(file_path: &Path, image: Image) {
    // compress the image
    let jpeg_data = ImageEncoder::new().encode(&image);

    // write the data directly to a file
    match std::fs::write(file_path, jpeg_data) {
        Ok(_) => {}
        Err(e) => panic!("Error writing file: {}", e),
    };
}

/// Reads an image from the given file path.
///
/// The method tries to read from any image format supported by the image crate.
///
/// # Arguments
///
/// * `file_path` - The path to the image.
///
// TODO: return sophus::TensorView
pub fn read_image_any(file_path: &Path) -> Image {
    // verify the file exists
    if !file_path.exists() {
        panic!("File does not exist: {}", file_path.to_str().unwrap());
    }

    // read the image
    let img: image::DynamicImage = match image::open(file_path) {
        Ok(img) => img,
        Err(e) => panic!("Error reading image: {}", e),
    };

    // return the image data
    let data = img.to_rgb8().to_vec();
    Image::new(ImageSize { width: img.width() as usize, height: img.height() as usize}, data)
}

#[cfg(test)]
mod tests {
    use std::fs;
    use std::path::Path;
    use tempfile::tempdir;

    use crate::io::functions::{read_image_any, read_image_jpeg, write_image_jpeg};

    #[test]
    fn read_jpeg() {
        let image_path = Path::new("tests/data/dog.jpeg");
        let image = read_image_jpeg(image_path);
        assert_eq!(image.image_size().width, 258);
        assert_eq!(image.image_size().height, 195);
    }

    #[test]
    fn read_any() {
        let image_path = Path::new("tests/data/dog.jpeg");
        let image = read_image_any(image_path);
        assert_eq!(image.image_size().width, 258);
        assert_eq!(image.image_size().height, 195);
    }

    #[test]
    fn read_write_jpeg() {
        let image_path_read = Path::new("tests/data/dog.jpeg");
        let tmp_dir = tempdir().unwrap();
        fs::create_dir_all(tmp_dir.path()).unwrap();
        let file_path = tmp_dir.path().join("dog.jpeg");
        let image_data = read_image_jpeg(image_path_read);
        write_image_jpeg(&file_path, image_data);
        let image_data_back = read_image_jpeg(&file_path);
        assert!(file_path.exists(), "File does not exist: {:?}", file_path);
        assert_eq!(image_data_back.image_size().width, 258);
        assert_eq!(image_data_back.image_size().height, 195);
        assert_eq!(image_data_back.num_channels(), 3);
    }
}
