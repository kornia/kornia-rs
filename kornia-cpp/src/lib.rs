#[cxx::bridge]
mod ffi {
    /// Image size in pixels
    ///
    /// Wraps kornia_image::ImageSize
    #[derive(Debug, Clone, Copy)]
    pub struct ImageSize {
        /// Width of the image in pixels
        pub width: usize,
        /// Height of the image in pixels
        pub height: usize,
    }

    // Image type declarations following OpenCV naming convention:
    // ImageU8C1  - Unsigned 8-bit,  1 Channel  (Grayscale)
    // ImageU8C3  - Unsigned 8-bit,  3 Channels (RGB)
    // ImageF32C3 - Float    32-bit, 3 Channels (RGB)
    
    extern "Rust" {
        // ============ ImageU8C1 (Grayscale u8) ============
        type ImageU8C1;
        fn image_u8c1_width(img: &ImageU8C1) -> usize;
        fn image_u8c1_height(img: &ImageU8C1) -> usize;
        fn image_u8c1_channels(img: &ImageU8C1) -> usize;
        fn image_u8c1_size(img: &ImageU8C1) -> ImageSize;
        fn image_u8c1_data(img: &ImageU8C1) -> &[u8];

        // ============ ImageU8C3 (RGB u8) ============
        type ImageU8C3;
        fn image_u8c3_width(img: &ImageU8C3) -> usize;
        fn image_u8c3_height(img: &ImageU8C3) -> usize;
        fn image_u8c3_channels(img: &ImageU8C3) -> usize;
        fn image_u8c3_size(img: &ImageU8C3) -> ImageSize;
        fn image_u8c3_data(img: &ImageU8C3) -> &[u8];

        // ============ ImageU8C4 (RGBA u8) ============
        type ImageU8C4;
        fn image_u8c4_width(img: &ImageU8C4) -> usize;
        fn image_u8c4_height(img: &ImageU8C4) -> usize;
        fn image_u8c4_channels(img: &ImageU8C4) -> usize;
        fn image_u8c4_size(img: &ImageU8C4) -> ImageSize;
        fn image_u8c4_data(img: &ImageU8C4) -> &[u8];

        // ============ ImageF32C1 (Grayscale f32) ============
        type ImageF32C1;
        fn image_f32c1_width(img: &ImageF32C1) -> usize;
        fn image_f32c1_height(img: &ImageF32C1) -> usize;
        fn image_f32c1_channels(img: &ImageF32C1) -> usize;
        fn image_f32c1_size(img: &ImageF32C1) -> ImageSize;
        fn image_f32c1_data(img: &ImageF32C1) -> &[f32];

        // ============ ImageF32C3 (RGB f32) ============
        type ImageF32C3;
        fn image_f32c3_width(img: &ImageF32C3) -> usize;
        fn image_f32c3_height(img: &ImageF32C3) -> usize;
        fn image_f32c3_channels(img: &ImageF32C3) -> usize;
        fn image_f32c3_size(img: &ImageF32C3) -> ImageSize;
        fn image_f32c3_data(img: &ImageF32C3) -> &[f32];

        // ============ ImageF32C4 (RGBA f32) ============
        type ImageF32C4;
        fn image_f32c4_width(img: &ImageF32C4) -> usize;
        fn image_f32c4_height(img: &ImageF32C4) -> usize;
        fn image_f32c4_channels(img: &ImageF32C4) -> usize;
        fn image_f32c4_size(img: &ImageF32C4) -> ImageSize;
        fn image_f32c4_data(img: &ImageF32C4) -> &[f32];

        // I/O functions
        
        /// Read a grayscale JPEG image from file path
        ///
        /// # Arguments
        ///
        /// * `file_path` - Path to the JPEG file
        ///
        /// # Returns
        ///
        /// ImageU8C1 containing grayscale data
        ///
        /// # Errors
        ///
        /// Throws exception if file cannot be read or is invalid
        fn read_jpeg_mono8(file_path: &str) -> Result<Box<ImageU8C1>>;

        /// Read an RGB JPEG image from file path
        ///
        /// # Arguments
        ///
        /// * `file_path` - Path to the JPEG file
        ///
        /// # Returns
        ///
        /// ImageU8C3 containing RGB data
        ///
        /// # Errors
        ///
        /// Throws exception if file cannot be read or is invalid
        fn read_jpeg_rgb8(file_path: &str) -> Result<Box<ImageU8C3>>;

        // Image creation functions
        
        /// Create a new ImageU8C3 with specified dimensions, filled with a value
        ///
        /// # Arguments
        ///
        /// * `width` - Image width in pixels
        /// * `height` - Image height in pixels
        /// * `value` - Initial value for all pixels
        ///
        /// # Returns
        ///
        /// ImageU8C3 filled with the specified value
        fn image_u8c3_new(width: usize, height: usize, value: u8) -> Box<ImageU8C3>;

        /// Create a new ImageU8C1 with specified dimensions, filled with a value
        ///
        /// # Arguments
        ///
        /// * `width` - Image width in pixels
        /// * `height` - Image height in pixels
        /// * `value` - Initial value for all pixels
        ///
        /// # Returns
        ///
        /// ImageU8C1 filled with the specified value
        fn image_u8c1_new(width: usize, height: usize, value: u8) -> Box<ImageU8C1>;

        /// Create a new ImageU8C3 from existing data
        ///
        /// # Arguments
        ///
        /// * `width` - Image width in pixels
        /// * `height` - Image height in pixels
        /// * `data` - Slice containing pixel data (length must be width * height * 3)
        ///
        /// # Returns
        ///
        /// ImageU8C3 containing the provided data
        fn image_u8c3_from_data(width: usize, height: usize, data: &[u8]) -> Box<ImageU8C3>;

        /// Create a new ImageU8C1 from existing data
        ///
        /// # Arguments
        ///
        /// * `width` - Image width in pixels
        /// * `height` - Image height in pixels
        /// * `data` - Slice containing pixel data (length must be width * height)
        ///
        /// # Returns
        ///
        /// ImageU8C1 containing the provided data
        fn image_u8c1_from_data(width: usize, height: usize, data: &[u8]) -> Box<ImageU8C1>;
    }
}

use kornia_image::allocator::CpuAllocator;
use kornia_io::jpeg;

// Conversion traits to seamlessly use kornia_image::ImageSize
impl From<kornia_image::ImageSize> for ffi::ImageSize {
    fn from(size: kornia_image::ImageSize) -> Self {
        ffi::ImageSize {
            width: size.width,
            height: size.height,
        }
    }
}

impl From<ffi::ImageSize> for kornia_image::ImageSize {
    fn from(size: ffi::ImageSize) -> Self {
        kornia_image::ImageSize {
            width: size.width,
            height: size.height,
        }
    }
}

/// Macro to define image wrapper types and their accessor functions
/// 
/// This reduces boilerplate for wrapping kornia_image::Image<T, C> types.
/// All methods directly delegate to the underlying kornia_image::Image.
///
/// Usage: define_image_type!(ImageU8_3, image_u8_3, u8, 3);
/// 
/// Args: (TypeName, fn_prefix, dtype, channels)
macro_rules! define_image_type {
    ($wrapper:ident, $prefix:ident, $dtype:ty, $ch:expr) => {
        pub struct $wrapper(kornia_image::Image<$dtype, $ch, CpuAllocator>);

        ::paste::paste! {
            fn [<$prefix _width>](img: &$wrapper) -> usize { img.0.width() }
            fn [<$prefix _height>](img: &$wrapper) -> usize { img.0.height() }
            fn [<$prefix _channels>](_img: &$wrapper) -> usize { $ch }
            fn [<$prefix _size>](img: &$wrapper) -> ffi::ImageSize { img.0.size().into() }
            fn [<$prefix _data>](img: &$wrapper) -> &[$dtype] { img.0.as_slice() }
        }
    };
}

// Define all supported image types using the macro
// Format: define_image_type!(TypeName, fn_prefix, dtype, channels);
//
// Naming convention (following OpenCV):
//   ImageU8C1  = Unsigned 8-bit,  1 Channel  (Grayscale)
//   ImageU8C3  = Unsigned 8-bit,  3 Channels (RGB)
//   ImageF32C3 = Float    32-bit, 3 Channels (RGB)

// u8 images (most common for I/O)
define_image_type!(ImageU8C1, image_u8c1, u8, 1);   // Grayscale u8
define_image_type!(ImageU8C3, image_u8c3, u8, 3);   // RGB u8
define_image_type!(ImageU8C4, image_u8c4, u8, 4);   // RGBA u8

// f32 images (common for processing/ML)
define_image_type!(ImageF32C1, image_f32c1, f32, 1); // Grayscale f32
define_image_type!(ImageF32C3, image_f32c3, f32, 3); // RGB f32
define_image_type!(ImageF32C4, image_f32c4, f32, 4); // RGBA f32

// Future: Add more types as needed
// define_image_type!(ImageU16C1, image_u16c1, u16, 1);
// define_image_type!(ImageU16C3, image_u16c3, u16, 3);
// define_image_type!(ImageF64C3, image_f64c3, f64, 3);

// I/O function implementations

/// Read a grayscale JPEG image from file path
fn read_jpeg_mono8(file_path: &str) -> Result<Box<ImageU8C1>, Box<dyn std::error::Error>> {
    let image = jpeg::read_image_jpeg_mono8(file_path)?;
    Ok(Box::new(ImageU8C1(image)))
}

/// Read an RGB JPEG image from file path
fn read_jpeg_rgb8(file_path: &str) -> Result<Box<ImageU8C3>, Box<dyn std::error::Error>> {
    let image = jpeg::read_image_jpeg_rgb8(file_path)?;
    Ok(Box::new(ImageU8C3(image)))
}

// Image creation functions

/// Create a new ImageU8C3 with specified dimensions, filled with a value
fn image_u8c3_new(width: usize, height: usize, value: u8) -> Box<ImageU8C3> {
    let size = kornia_image::ImageSize { width, height };
    let alloc = CpuAllocator::default();
    let image = kornia_image::Image::<u8, 3, CpuAllocator>::from_size_val(size, value, alloc)
        .expect("Failed to create image");
    Box::new(ImageU8C3(image))
}

/// Create a new ImageU8C1 with specified dimensions, filled with a value
fn image_u8c1_new(width: usize, height: usize, value: u8) -> Box<ImageU8C1> {
    let size = kornia_image::ImageSize { width, height };
    let alloc = CpuAllocator::default();
    let image = kornia_image::Image::<u8, 1, CpuAllocator>::from_size_val(size, value, alloc)
        .expect("Failed to create image");
    Box::new(ImageU8C1(image))
}

/// Create a new ImageU8C3 from existing data
fn image_u8c3_from_data(width: usize, height: usize, data: &[u8]) -> Box<ImageU8C3> {
    let size = kornia_image::ImageSize { width, height };
    let expected_len = width * height * 3;
    if data.len() != expected_len {
        panic!(
            "Data length mismatch: expected {} bytes, got {}",
            expected_len,
            data.len()
        );
    }
    let alloc = CpuAllocator::default();
    let image = kornia_image::Image::<u8, 3, CpuAllocator>::from_size_slice(size, data, alloc)
        .expect("Failed to create image from data");
    Box::new(ImageU8C3(image))
}

/// Create a new ImageU8C1 from existing data
fn image_u8c1_from_data(width: usize, height: usize, data: &[u8]) -> Box<ImageU8C1> {
    let size = kornia_image::ImageSize { width, height };
    let expected_len = width * height;
    if data.len() != expected_len {
        panic!(
            "Data length mismatch: expected {} bytes, got {}",
            expected_len,
            data.len()
        );
    }
    let alloc = CpuAllocator::default();
    let image = kornia_image::Image::<u8, 1, CpuAllocator>::from_size_slice(size, data, alloc)
        .expect("Failed to create image from data");
    Box::new(ImageU8C1(image))
}

