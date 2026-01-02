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
    //
    // Note: cxx::bridge doesn't support macros in extern "Rust" blocks,
    // so types must be declared explicitly. However, implementations use
    // macros to reduce boilerplate.

    extern "Rust" {
        // ============ ImageU8C1 (Grayscale u8) ============
        type ImageU8C1;
        fn image_u8c1_width(img: &ImageU8C1) -> usize;
        fn image_u8c1_height(img: &ImageU8C1) -> usize;
        fn image_u8c1_channels(img: &ImageU8C1) -> usize;
        fn image_u8c1_size(img: &ImageU8C1) -> ImageSize;
        fn image_u8c1_data(img: &ImageU8C1) -> &[u8];
        fn image_u8c1_new(width: usize, height: usize, value: u8) -> Result<Box<ImageU8C1>>;
        fn image_u8c1_from_data(width: usize, height: usize, data: &[u8])
            -> Result<Box<ImageU8C1>>;

        // ============ ImageU8C3 (RGB u8) ============
        type ImageU8C3;
        fn image_u8c3_width(img: &ImageU8C3) -> usize;
        fn image_u8c3_height(img: &ImageU8C3) -> usize;
        fn image_u8c3_channels(img: &ImageU8C3) -> usize;
        fn image_u8c3_size(img: &ImageU8C3) -> ImageSize;
        fn image_u8c3_data(img: &ImageU8C3) -> &[u8];
        fn image_u8c3_new(width: usize, height: usize, value: u8) -> Result<Box<ImageU8C3>>;
        fn image_u8c3_from_data(width: usize, height: usize, data: &[u8])
            -> Result<Box<ImageU8C3>>;

        // ============ ImageU8C4 (RGBA u8) ============
        type ImageU8C4;
        fn image_u8c4_width(img: &ImageU8C4) -> usize;
        fn image_u8c4_height(img: &ImageU8C4) -> usize;
        fn image_u8c4_channels(img: &ImageU8C4) -> usize;
        fn image_u8c4_size(img: &ImageU8C4) -> ImageSize;
        fn image_u8c4_data(img: &ImageU8C4) -> &[u8];
        fn image_u8c4_new(width: usize, height: usize, value: u8) -> Result<Box<ImageU8C4>>;
        fn image_u8c4_from_data(width: usize, height: usize, data: &[u8])
            -> Result<Box<ImageU8C4>>;

        // ============ ImageF32C1 (Grayscale f32) ============
        type ImageF32C1;
        fn image_f32c1_width(img: &ImageF32C1) -> usize;
        fn image_f32c1_height(img: &ImageF32C1) -> usize;
        fn image_f32c1_channels(img: &ImageF32C1) -> usize;
        fn image_f32c1_size(img: &ImageF32C1) -> ImageSize;
        fn image_f32c1_data(img: &ImageF32C1) -> &[f32];
        fn image_f32c1_new(width: usize, height: usize, value: f32) -> Result<Box<ImageF32C1>>;
        fn image_f32c1_from_data(
            width: usize,
            height: usize,
            data: &[f32],
        ) -> Result<Box<ImageF32C1>>;

        // ============ ImageF32C3 (RGB f32) ============
        type ImageF32C3;
        fn image_f32c3_width(img: &ImageF32C3) -> usize;
        fn image_f32c3_height(img: &ImageF32C3) -> usize;
        fn image_f32c3_channels(img: &ImageF32C3) -> usize;
        fn image_f32c3_size(img: &ImageF32C3) -> ImageSize;
        fn image_f32c3_data(img: &ImageF32C3) -> &[f32];
        fn image_f32c3_new(width: usize, height: usize, value: f32) -> Result<Box<ImageF32C3>>;
        fn image_f32c3_from_data(
            width: usize,
            height: usize,
            data: &[f32],
        ) -> Result<Box<ImageF32C3>>;

        // ============ ImageF32C4 (RGBA f32) ============
        type ImageF32C4;
        fn image_f32c4_width(img: &ImageF32C4) -> usize;
        fn image_f32c4_height(img: &ImageF32C4) -> usize;
        fn image_f32c4_channels(img: &ImageF32C4) -> usize;
        fn image_f32c4_size(img: &ImageF32C4) -> ImageSize;
        fn image_f32c4_data(img: &ImageF32C4) -> &[f32];
        fn image_f32c4_new(width: usize, height: usize, value: f32) -> Result<Box<ImageF32C4>>;
        fn image_f32c4_from_data(
            width: usize,
            height: usize,
            data: &[f32],
        ) -> Result<Box<ImageF32C4>>;

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
        fn read_image_jpeg_mono8(file_path: &str) -> Result<Box<ImageU8C1>>;

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
        fn read_image_jpeg_rgb8(file_path: &str) -> Result<Box<ImageU8C3>>;

        /// Encode an RGB u8 image to JPEG bytes
        ///
        /// # Arguments
        ///
        /// * `image` - The RGB image to encode
        /// * `quality` - JPEG quality (0-100, where 100 is highest quality)
        /// * `buffer` - Output buffer to write the JPEG bytes into
        ///
        /// # Note
        ///
        /// The caller is responsible for clearing the buffer if needed.
        ///
        /// # Errors
        ///
        /// Throws exception if encoding fails
        fn encode_image_jpeg_rgb8(
            image: &ImageU8C3,
            quality: u8,
            buffer: &mut Vec<u8>,
        ) -> Result<()>;

        /// Encode a BGRA u8 image to JPEG bytes
        ///
        /// # Arguments
        ///
        /// * `image` - The BGRA image to encode
        /// * `quality` - JPEG quality (0-100, where 100 is highest quality)
        /// * `buffer` - Output buffer to write the JPEG bytes into
        ///
        /// # Note
        ///
        /// The caller is responsible for clearing the buffer if needed.
        /// BGRA format is common in graphics APIs like DirectX and Unreal Engine.
        ///
        /// # Errors
        ///
        /// Throws exception if encoding fails
        fn encode_image_jpeg_bgra8(
            image: &ImageU8C4,
            quality: u8,
            buffer: &mut Vec<u8>,
        ) -> Result<()>;

        /// Decode JPEG bytes to RGB u8 image
        ///
        /// # Arguments
        ///
        /// * `jpeg_bytes` - JPEG-encoded bytes (borrows via slice)
        ///
        /// # Returns
        ///
        /// ImageU8C3 containing decoded RGB data
        ///
        /// # Errors
        ///
        /// Throws exception if decoding fails
        fn decode_image_jpeg_rgb8(jpeg_bytes: &[u8]) -> Result<Box<ImageU8C3>>;

        fn version() -> &'static str;
    }
}

use kornia_image::allocator::CpuAllocator;
use kornia_image::ImageError;
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
/// Generates:
/// - Wrapper struct
/// - Accessor functions (width, height, channels, size, data)
/// - Constructor functions (_new, _from_data)
///
/// Usage: define_image_type!(ImageU8_3, image_u8_3, u8, 3);
///
/// Args: (TypeName, fn_prefix, dtype, channels)
macro_rules! define_image_type {
    ($wrapper:ident, $prefix:ident, $dtype:ty, $ch:expr) => {
        pub struct $wrapper(kornia_image::Image<$dtype, $ch, CpuAllocator>);

        ::paste::paste! {
            // Accessor functions
            fn [<$prefix _width>](img: &$wrapper) -> usize { img.0.width() }
            fn [<$prefix _height>](img: &$wrapper) -> usize { img.0.height() }
            fn [<$prefix _channels>](_img: &$wrapper) -> usize { $ch }
            fn [<$prefix _size>](img: &$wrapper) -> ffi::ImageSize { img.0.size().into() }
            fn [<$prefix _data>](img: &$wrapper) -> &[$dtype] { img.0.as_slice() }

            // Constructor: from size and fill value
            fn [<$prefix _new>](width: usize, height: usize, value: $dtype) -> Result<Box<$wrapper>, ImageError> {
                let size = kornia_image::ImageSize { width, height };
                let alloc = CpuAllocator::default();
                let image = kornia_image::Image::<$dtype, $ch, CpuAllocator>::from_size_val(size, value, alloc)?;
                Ok(Box::new($wrapper(image)))
            }

            // Constructor: from existing data
            fn [<$prefix _from_data>](width: usize, height: usize, data: &[$dtype]) -> Result<Box<$wrapper>, ImageError> {
                let size = kornia_image::ImageSize { width, height };
                let expected_len = width * height * $ch;
                if data.len() != expected_len {
                    return Err(ImageError::InvalidChannelShape(data.len(), expected_len));
                }
                let alloc = CpuAllocator::default();
                let image = kornia_image::Image::<$dtype, $ch, CpuAllocator>::from_size_slice(size, data, alloc)?;
                Ok(Box::new($wrapper(image)))
            }
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
define_image_type!(ImageU8C1, image_u8c1, u8, 1); // Grayscale u8
define_image_type!(ImageU8C3, image_u8c3, u8, 3); // RGB u8
define_image_type!(ImageU8C4, image_u8c4, u8, 4); // RGBA u8

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
fn read_image_jpeg_mono8(file_path: &str) -> Result<Box<ImageU8C1>, Box<dyn std::error::Error>> {
    let image = jpeg::read_image_jpeg_mono8(file_path)?;
    Ok(Box::new(ImageU8C1(image.into_inner())))
}

/// Read an RGB JPEG image from file path
fn read_image_jpeg_rgb8(file_path: &str) -> Result<Box<ImageU8C3>, Box<dyn std::error::Error>> {
    let image = jpeg::read_image_jpeg_rgb8(file_path)?;
    Ok(Box::new(ImageU8C3(image.into_inner())))
}

/// Encode an RGB u8 image to JPEG bytes
fn encode_image_jpeg_rgb8(
    image: &ImageU8C3,
    quality: u8,
    buffer: &mut Vec<u8>,
) -> Result<(), Box<dyn std::error::Error>> {
    jpeg::encode_image_jpeg_rgb8(&image.0, quality, buffer)?;
    Ok(())
}

/// Encode a BGRA u8 image to JPEG bytes
fn encode_image_jpeg_bgra8(
    image: &ImageU8C4,
    quality: u8,
    buffer: &mut Vec<u8>,
) -> Result<(), Box<dyn std::error::Error>> {
    jpeg::encode_image_jpeg_bgra8(&image.0, quality, buffer)?;
    Ok(())
}

/// Decode JPEG bytes to RGB u8 image (zero-copy via slice)
fn decode_image_jpeg_rgb8(jpeg_bytes: &[u8]) -> Result<Box<ImageU8C3>, Box<dyn std::error::Error>> {
    // First, get image info to create properly sized image
    let layout = jpeg::decode_image_jpeg_layout(jpeg_bytes)?;

    if layout.channels != 3 {
        return Err(format!(
            "Expected RGB (3 channels), got {} channels",
            layout.channels
        )
        .into());
    }

    // Create output image
    let mut image = kornia_image::Image::<u8, 3, CpuAllocator>::from_size_val(
        layout.image_size,
        0,
        CpuAllocator,
    )?;

    // Decode into it (zero-copy from slice)
    jpeg::decode_image_jpeg_rgb8(jpeg_bytes, &mut image)?;

    Ok(Box::new(ImageU8C3(image)))
}

/// Get library version string
fn version() -> &'static str {
    env!("CARGO_PKG_VERSION")
}
