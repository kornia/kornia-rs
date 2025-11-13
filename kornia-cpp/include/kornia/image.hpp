#pragma once

#include "kornia-cpp/src/lib.rs.h"
#include <cstddef>
#include <cstdint>
#include <vector>

// Check for C++20 std::span support
#if __cplusplus >= 202002L && __has_include(<span>)
#include <span>
#define KORNIA_HAS_STD_SPAN 1
#endif

namespace kornia {
namespace image {

/// @brief Image size in pixels
///
/// Matches kornia_image::ImageSize
using ImageSize = ::ImageSize;

/// @brief Zero-copy buffer for encoded image data
///
/// Generic buffer for all image encoding formats (JPEG, PNG, WebP, TIFF, etc.).
/// Wraps the underlying Rust buffer to provide zero-copy access without
/// exposing Rust types to user code. Data stays in Rust-managed memory.
///
/// @note Reuse the same buffer across different encoders to minimize allocations.
///
/// @example
/// @code
/// kornia::image::ImageBuffer buffer;
/// for (const auto& image : images) {
///     buffer.clear();
///     kornia::io::encode_jpeg_rgb8(image, 95, buffer);
///     send_network(buffer.data(), buffer.size());
///
///     buffer.clear();
///     kornia::io::encode_png_rgb8(image, buffer);  // Same buffer!
/// }
/// @endcode
class ImageBuffer {
  public:
    ImageBuffer() = default;

    /// @brief Get raw pointer to encoded data (zero-copy)
    /// @return Pointer to encoded bytes
    const uint8_t* data() const {
        return rust_buffer_.data();
    }

    /// @brief Get size of encoded data in bytes
    /// @return Size in bytes
    size_t size() const {
        return rust_buffer_.size();
    }

    /// @brief Check if buffer is empty
    /// @return true if empty
    bool empty() const {
        return rust_buffer_.empty();
    }

    /// @brief Explicitly convert to std::vector (performs copy)
    /// @return std::vector containing a copy of the encoded data
    /// @note This performs a copy. Use data()/size() for zero-copy access.
    std::vector<uint8_t> to_vector() const {
        return {rust_buffer_.begin(), rust_buffer_.end()};
    }

    /// @brief Clear the buffer (retains capacity for reuse)
    void clear() {
        rust_buffer_.clear();
    }

    /// @brief Get access to internal rust::Vec (for I/O functions)
    /// @note Internal use only - not part of public API
    rust::Vec<uint8_t>& rust_vec() {
        return rust_buffer_;
    }

  private:
    rust::Vec<uint8_t> rust_buffer_; // Hidden implementation detail
};

// Macro to generate C++ wrapper classes for image types.
// All methods delegate to Rust FFI functions (zero-copy via rust::Slice).
// Naming follows OpenCV convention: ImageU8C3 = Unsigned 8-bit, 3 Channels
//
// Thread safety: Concurrent reads are safe. Do not move/destroy while other threads access.
#define KORNIA_DEFINE_IMAGE_WRAPPER(CppClass, RustType, FnPrefix, DataType)                        \
    class CppClass {                                                                               \
      public:                                                                                      \
        CppClass(rust::Box<::RustType> img) : img_(std::move(img)) {                               \
        }                                                                                          \
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
        const ::RustType& inner() const {                                                          \
            return *img_;                                                                          \
        }                                                                                          \
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

} // namespace image
} // namespace kornia
