use crate::{
    allocator::ImageAllocator,
    error::ImageError,
    image::{Image, ImageSize},
};
use std::ops::{Deref, DerefMut};

/// Macro to define a color space wrapper type with explicit bit depth
macro_rules! define_color_space {
    ($name:ident, $type:ty, $channels:expr, $doc:expr) => {
        #[doc = $doc]
        ///
        /// This is a zero-cost wrapper that provides compile-time type safety.
        #[repr(transparent)]
        pub struct $name<A: ImageAllocator>(pub Image<$type, $channels, A>);

        impl<A: ImageAllocator> $name<A> {
            #[doc = concat!("Create ", stringify!($name), " image from size and data")]
            pub fn from_size_vec(
                size: ImageSize,
                data: Vec<$type>,
                alloc: A,
            ) -> Result<Self, ImageError> {
                Ok(Self(Image::new(size, data, alloc)?))
            }

            #[doc = concat!("Create ", stringify!($name), " image from size with default value")]
            pub fn from_size_val(
                size: ImageSize,
                val: $type,
                alloc: A,
            ) -> Result<Self, ImageError> {
                Ok(Self(Image::from_size_val(size, val, alloc)?))
            }

            /// Unwrap into the underlying Image
            pub fn into_inner(self) -> Image<$type, $channels, A> {
                self.0
            }

            /// Get a reference to the underlying Image
            pub fn as_image(&self) -> &Image<$type, $channels, A> {
                &self.0
            }

            /// Get a mutable reference to the underlying Image
            pub fn as_image_mut(&mut self) -> &mut Image<$type, $channels, A> {
                &mut self.0
            }
        }

        impl<A: ImageAllocator> Deref for $name<A> {
            type Target = Image<$type, $channels, A>;
            fn deref(&self) -> &Self::Target {
                &self.0
            }
        }

        impl<A: ImageAllocator> DerefMut for $name<A> {
            fn deref_mut(&mut self) -> &mut Self::Target {
                &mut self.0
            }
        }

        impl<A: ImageAllocator> AsRef<Image<$type, $channels, A>> for $name<A> {
            fn as_ref(&self) -> &Image<$type, $channels, A> {
                &self.0
            }
        }
    };
}

// Define explicit color space types with bit depths
define_color_space!(
    Rgb8,
    u8,
    3,
    "RGB color space with 8-bit unsigned integer channels"
);
define_color_space!(
    Rgb16,
    u16,
    3,
    "RGB color space with 16-bit unsigned integer channels"
);
define_color_space!(
    Rgbf32,
    f32,
    3,
    "RGB color space with 32-bit floating point channels"
);
define_color_space!(
    Rgbf64,
    f64,
    3,
    "RGB color space with 64-bit floating point channels"
);

define_color_space!(
    Bgr8,
    u8,
    3,
    "BGR color space with 8-bit unsigned integer channels"
);
define_color_space!(
    Bgr16,
    u16,
    3,
    "BGR color space with 16-bit unsigned integer channels"
);
define_color_space!(
    Bgrf32,
    f32,
    3,
    "BGR color space with 32-bit floating point channels"
);
define_color_space!(
    Bgrf64,
    f64,
    3,
    "BGR color space with 64-bit floating point channels"
);

define_color_space!(
    Gray8,
    u8,
    1,
    "Grayscale with 8-bit unsigned integer channels"
);
define_color_space!(
    Gray16,
    u16,
    1,
    "Grayscale with 16-bit unsigned integer channels"
);
define_color_space!(
    Grayf32,
    f32,
    1,
    "Grayscale with 32-bit floating point channels"
);
define_color_space!(
    Grayf64,
    f64,
    1,
    "Grayscale with 64-bit floating point channels"
);

define_color_space!(
    Rgba8,
    u8,
    4,
    "RGBA color space with 8-bit unsigned integer channels"
);
define_color_space!(
    Rgba16,
    u16,
    4,
    "RGBA color space with 16-bit unsigned integer channels"
);
define_color_space!(
    Rgbaf32,
    f32,
    4,
    "RGBA color space with 32-bit floating point channels"
);
define_color_space!(
    Rgbaf64,
    f64,
    4,
    "RGBA color space with 64-bit floating point channels"
);

define_color_space!(
    Bgra8,
    u8,
    4,
    "BGRA color space with 8-bit unsigned integer channels"
);
define_color_space!(
    Bgra16,
    u16,
    4,
    "BGRA color space with 16-bit unsigned integer channels"
);
define_color_space!(
    Bgraf32,
    f32,
    4,
    "BGRA color space with 32-bit floating point channels"
);
define_color_space!(
    Bgraf64,
    f64,
    4,
    "BGRA color space with 64-bit floating point channels"
);

define_color_space!(
    Hsvf32,
    f32,
    3,
    "HSV color space with 32-bit floating point channels"
);
define_color_space!(
    Hsvf64,
    f64,
    3,
    "HSV color space with 64-bit floating point channels"
);
