#[cxx::bridge]
mod ffi {
    /// Simple struct to represent an image result
    pub struct ImageResult {
        pub data: Vec<u8>,
        pub width: usize,
        pub height: usize,
        pub channels: usize,
        pub success: bool,
        pub error_message: String,
    }

    extern "Rust" {
        /// Read a grayscale JPEG image from file path
        ///
        /// # Arguments
        ///
        /// * `file_path` - Path to the JPEG file
        ///
        /// # Returns
        ///
        /// ImageResult containing the image data, dimensions, and status
        fn read_jpeg_mono8(file_path: &str) -> ImageResult;

        /// Read an RGB JPEG image from file path
        ///
        /// # Arguments
        ///
        /// * `file_path` - Path to the JPEG file
        ///
        /// # Returns
        ///
        /// ImageResult containing the image data, dimensions, and status
        fn read_jpeg_rgb8(file_path: &str) -> ImageResult;
    }
}

use kornia_io::jpeg::{read_image_jpeg_mono8, read_image_jpeg_rgb8};

/// Implementation of the C++ callable function for grayscale images
fn read_jpeg_mono8(file_path: &str) -> ffi::ImageResult {
    match read_image_jpeg_mono8(file_path) {
        Ok(image) => {
            let size = image.size();
            let data = image.as_slice().to_vec();
            
            ffi::ImageResult {
                data,
                width: size.width,
                height: size.height,
                channels: 1,
                success: true,
                error_message: String::new(),
            }
        }
        Err(e) => ffi::ImageResult {
            data: Vec::new(),
            width: 0,
            height: 0,
            channels: 0,
            success: false,
            error_message: format!("Failed to read image: {}", e),
        },
    }
}

/// Implementation of the C++ callable function for RGB images
fn read_jpeg_rgb8(file_path: &str) -> ffi::ImageResult {
    match read_image_jpeg_rgb8(file_path) {
        Ok(image) => {
            let size = image.size();
            let data = image.as_slice().to_vec();
            
            ffi::ImageResult {
                data,
                width: size.width,
                height: size.height,
                channels: 3,
                success: true,
                error_message: String::new(),
            }
        }
        Err(e) => ffi::ImageResult {
            data: Vec::new(),
            width: 0,
            height: 0,
            channels: 0,
            success: false,
            error_message: format!("Failed to read image: {}", e),
        },
    }
}

