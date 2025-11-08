#pragma once

#include "kornia-cpp/src/lib.rs.h"

namespace kornia {

/// @brief Image size in pixels
///
/// Matches kornia_image::ImageSize
using ImageSize = ::ImageSize;

// Macro to generate C++ wrapper classes for image types.
// All methods delegate to Rust FFI functions (zero-copy via rust::Slice).
// Naming follows OpenCV convention: ImageU8C3 = Unsigned 8-bit, 3 Channels
#define KORNIA_DEFINE_IMAGE_WRAPPER(CppClass, RustType, FnPrefix, DataType)                        \
    class CppClass {                                                                               \
      public:                                                                                      \
        CppClass(rust::Box<::RustType> img) : img_(std::move(img)) {}                              \
                                                                                                   \
        size_t width() const { return FnPrefix##_width(*img_); }                                   \
        size_t height() const { return FnPrefix##_height(*img_); }                                 \
        size_t channels() const { return FnPrefix##_channels(*img_); }                             \
        ImageSize size() const { return FnPrefix##_size(*img_); }                                  \
        rust::Slice<const DataType> data() const { return FnPrefix##_data(*img_); }                \
                                                                                                   \
      private:                                                                                     \
        rust::Box<::RustType> img_;                                                                \
    }

// u8 image types (8-bit unsigned, common for I/O)
/// @brief Grayscale u8 image (1 channel). Wraps kornia_image::Image<u8, 1>.
KORNIA_DEFINE_IMAGE_WRAPPER(ImageU8C1, ImageU8C1, image_u8c1, uint8_t);

/// @brief RGB u8 image (3 channels). Wraps kornia_image::Image<u8, 3>.
KORNIA_DEFINE_IMAGE_WRAPPER(ImageU8C3, ImageU8C3, image_u8c3, uint8_t);

/// @brief RGBA u8 image (4 channels). Wraps kornia_image::Image<u8, 4>.
KORNIA_DEFINE_IMAGE_WRAPPER(ImageU8C4, ImageU8C4, image_u8c4, uint8_t);

// f32 image types (32-bit float, common for ML/processing)
/// @brief Grayscale f32 image (1 channel). Wraps kornia_image::Image<f32, 1>.
KORNIA_DEFINE_IMAGE_WRAPPER(ImageF32C1, ImageF32C1, image_f32c1, float);

/// @brief RGB f32 image (3 channels). Wraps kornia_image::Image<f32, 3>.
KORNIA_DEFINE_IMAGE_WRAPPER(ImageF32C3, ImageF32C3, image_f32c3, float);

/// @brief RGBA f32 image (4 channels). Wraps kornia_image::Image<f32, 4>.
KORNIA_DEFINE_IMAGE_WRAPPER(ImageF32C4, ImageF32C4, image_f32c4, float);

#undef KORNIA_DEFINE_IMAGE_WRAPPER

} // namespace kornia
