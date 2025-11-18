#pragma once

#include "kornia-cpp/src/lib.rs.h"
#include <vector>

namespace kornia {

/// @brief Image size in pixels
///
/// Matches kornia_image::ImageSize from Rust
using ImageSize = ::ImageSize;

/// @brief Macro to generate C++ wrapper classes for image types
///
/// Generates a complete wrapper class for a Rust image type with zero-copy semantics.
/// All methods delegate to Rust FFI functions via CXX bridge.
///
/// Naming convention: ImageU8C3 = Unsigned 8-bit, 3 Channels (OpenCV-style)
///
/// @param CppClass C++ class name (e.g., ImageU8C3)
/// @param RustType Rust type name from CXX bridge (e.g., ImageU8C3)
/// @param FnPrefix Function prefix for FFI calls (e.g., image_u8c3)
/// @param DataType C++ data type for pixel values (e.g., uint8_t, float)
///
/// Thread safety:
/// - Safe for concurrent reads from multiple threads
/// - Not safe to access after move or destruction
/// - Rust ownership ensures no data races
///
/// Memory management:
/// - Owned by Rust via rust::Box
/// - C++ holds non-owning view via rust::Slice
/// - No explicit cleanup needed (RAII via rust::Box destructor)
///
/// Zero-copy guarantee:
/// - data() returns rust::Slice pointing directly into Rust memory
/// - No data copies on FFI boundary
/// - Lifetime tied to wrapper object
#define KORNIA_DEFINE_IMAGE_WRAPPER(CppClass, RustType, FnPrefix, DataType)                        \
    class CppClass {                                                                               \
      public:                                                                                      \
        explicit CppClass(rust::Box<::RustType> img) : img_(std::move(img)) {                      \
        }                                                                                          \
                                                                                                   \
        CppClass(const CppClass&) = delete;                                                        \
        CppClass& operator=(const CppClass&) = delete;                                             \
        CppClass(CppClass&&) = default;                                                            \
        CppClass& operator=(CppClass&&) = default;                                                 \
                                                                                                   \
        size_t width() const {                                                                     \
            return FnPrefix##_width(*img_);                                                        \
        }                                                                                          \
        size_t height() const {                                                                    \
            return FnPrefix##_height(*img_);                                                       \
        }                                                                                          \
        size_t channels() const {                                                                  \
            return FnPrefix##_channels(*img_);                                                     \
        }                                                                                          \
        ImageSize size() const {                                                                   \
            return FnPrefix##_size(*img_);                                                         \
        }                                                                                          \
        rust::Slice<const DataType> data() const {                                                 \
            return FnPrefix##_data(*img_);                                                         \
        }                                                                                          \
                                                                                                   \
      private:                                                                                     \
        rust::Box<::RustType> img_;                                                                \
    }

/// @brief Macro to generate C++ wrapper classes with convenience constructors
///
/// Extended version that includes constructors for creating images from shape+value or shape+data.
/// Only use for image types that have corresponding FFI functions (_new and _from_data).
#define KORNIA_DEFINE_IMAGE_WRAPPER_WITH_CTORS(CppClass, RustType, FnPrefix, DataType)             \
    class CppClass {                                                                               \
      public:                                                                                      \
        explicit CppClass(rust::Box<::RustType> img) : img_(std::move(img)) {                      \
        }                                                                                          \
                                                                                                   \
        CppClass(size_t width, size_t height, DataType value)                                      \
            : img_(::FnPrefix##_new(width, height, value)) {                                       \
        }                                                                                          \
                                                                                                   \
        CppClass(size_t width, size_t height, const std::vector<DataType>& data)                   \
            : img_(::FnPrefix##_from_data(                                                         \
                  width, height, rust::Slice<const DataType>(data.data(), data.size()))) {         \
        }                                                                                          \
                                                                                                   \
        CppClass(const CppClass&) = delete;                                                        \
        CppClass& operator=(const CppClass&) = delete;                                             \
        CppClass(CppClass&&) = default;                                                            \
        CppClass& operator=(CppClass&&) = default;                                                 \
                                                                                                   \
        size_t width() const {                                                                     \
            return FnPrefix##_width(*img_);                                                        \
        }                                                                                          \
        size_t height() const {                                                                    \
            return FnPrefix##_height(*img_);                                                       \
        }                                                                                          \
        size_t channels() const {                                                                  \
            return FnPrefix##_channels(*img_);                                                     \
        }                                                                                          \
        ImageSize size() const {                                                                   \
            return FnPrefix##_size(*img_);                                                         \
        }                                                                                          \
        rust::Slice<const DataType> data() const {                                                 \
            return FnPrefix##_data(*img_);                                                         \
        }                                                                                          \
                                                                                                   \
      private:                                                                                     \
        rust::Box<::RustType> img_;                                                                \
    }

// u8 image types (8-bit unsigned, common for I/O)
/// @brief Grayscale u8 image (1 channel). Wraps kornia_image::Image<u8, 1>.
///
/// Thread safety: Safe for concurrent reads. Do not access after move or destruction.
/// Memory: Managed by Rust, zero-copy data access via rust::Slice.
///
/// Constructors:
/// - ImageU8C1(width, height, value) - Create filled with value
/// - ImageU8C1(width, height, data) - Create from std::vector<uint8_t>
KORNIA_DEFINE_IMAGE_WRAPPER_WITH_CTORS(ImageU8C1, ImageU8C1, image_u8c1, uint8_t);

/// @brief RGB u8 image (3 channels). Wraps kornia_image::Image<u8, 3>.
///
/// Thread safety: Safe for concurrent reads. Do not access after move or destruction.
/// Memory: Managed by Rust, zero-copy data access via rust::Slice.
///
/// Constructors:
/// - ImageU8C3(width, height, value) - Create filled with value
/// - ImageU8C3(width, height, data) - Create from std::vector<uint8_t>
KORNIA_DEFINE_IMAGE_WRAPPER_WITH_CTORS(ImageU8C3, ImageU8C3, image_u8c3, uint8_t);

/// @brief RGBA u8 image (4 channels). Wraps kornia_image::Image<u8, 4>.
///
/// Thread safety: Safe for concurrent reads. Do not access after move or destruction.
/// Memory: Managed by Rust, zero-copy data access via rust::Slice.
KORNIA_DEFINE_IMAGE_WRAPPER(ImageU8C4, ImageU8C4, image_u8c4, uint8_t);

// f32 image types (32-bit float, common for ML/processing)
/// @brief Grayscale f32 image (1 channel). Wraps kornia_image::Image<f32, 1>.
///
/// Thread safety: Safe for concurrent reads. Do not access after move or destruction.
/// Memory: Managed by Rust, zero-copy data access via rust::Slice.
KORNIA_DEFINE_IMAGE_WRAPPER(ImageF32C1, ImageF32C1, image_f32c1, float);

/// @brief RGB f32 image (3 channels). Wraps kornia_image::Image<f32, 3>.
///
/// Thread safety: Safe for concurrent reads. Do not access after move or destruction.
/// Memory: Managed by Rust, zero-copy data access via rust::Slice.
KORNIA_DEFINE_IMAGE_WRAPPER(ImageF32C3, ImageF32C3, image_f32c3, float);

/// @brief RGBA f32 image (4 channels). Wraps kornia_image::Image<f32, 4>.
///
/// Thread safety: Safe for concurrent reads. Do not access after move or destruction.
/// Memory: Managed by Rust, zero-copy data access via rust::Slice.
KORNIA_DEFINE_IMAGE_WRAPPER(ImageF32C4, ImageF32C4, image_f32c4, float);

#undef KORNIA_DEFINE_IMAGE_WRAPPER
#undef KORNIA_DEFINE_IMAGE_WRAPPER_WITH_CTORS

} // namespace kornia
