#pragma once

#include "kornia/detail/lib.rs.h"
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
/// Matches kornia_image::ImageSize from Rust
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

/// @brief Traits-based image wrapper with template support
///
/// Uses a traits pattern to map (T, C) to Rust FFI types and functions.
/// This enables a single template implementation shared by all image types.
///
/// Naming convention: Image<uint8_t, 3> = Unsigned 8-bit, 3 Channels (OpenCV-style)
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
/// - to_vec() returns std::vector<T> (owned copy of the data)
/// - No data copies on FFI boundary for data()
/// - Lifetime tied to wrapper object

// Forward declaration of traits
template <typename T, size_t C> struct ImageTraits;

// X-Macros pattern: Single source of truth for all image types
// Define the list once, expand it multiple times for different purposes
#define KORNIA_IMAGE_TYPES                                                                         \
    KORNIA_IMAGE_TYPE(uint8_t, ImageU8C1, image_u8c1, 1)                                           \
    KORNIA_IMAGE_TYPE(uint8_t, ImageU8C3, image_u8c3, 3)                                           \
    KORNIA_IMAGE_TYPE(uint8_t, ImageU8C4, image_u8c4, 4)                                           \
    KORNIA_IMAGE_TYPE(float, ImageF32C1, image_f32c1, 1)                                           \
    KORNIA_IMAGE_TYPE(float, ImageF32C3, image_f32c3, 3)                                           \
    KORNIA_IMAGE_TYPE(float, ImageF32C4, image_f32c4, 4)

// First expansion: Generate traits (before Image template)
#define KORNIA_IMAGE_TYPE(CppType, RustType, FnPrefix, Channels)                                   \
    template <> struct ImageTraits<CppType, Channels> {                                            \
        using RustType_ = ::RustType;                                                              \
        static size_t width(const RustType_& img) {                                                \
            return FnPrefix##_width(img);                                                          \
        }                                                                                          \
        static size_t height(const RustType_& img) {                                               \
            return FnPrefix##_height(img);                                                         \
        }                                                                                          \
        static size_t channels(const RustType_&) {                                                 \
            return Channels;                                                                       \
        }                                                                                          \
        static ImageSize size(const RustType_& img) {                                              \
            return FnPrefix##_size(img);                                                           \
        }                                                                                          \
        static rust::Slice<const CppType> data(const RustType_& img) {                             \
            return FnPrefix##_data(img);                                                           \
        }                                                                                          \
        static rust::Box<RustType_> new_image(size_t w, size_t h, CppType v) {                     \
            return FnPrefix##_new(w, h, v);                                                        \
        }                                                                                          \
        static rust::Box<RustType_> from_data(size_t w, size_t h, rust::Slice<const CppType> d) {  \
            return FnPrefix##_from_data(w, h, d);                                                  \
        }                                                                                          \
    };

KORNIA_IMAGE_TYPES
#undef KORNIA_IMAGE_TYPE

/// @brief Generic image wrapper template
///
/// Usage: kornia::image::Image<uint8_t, 3> for RGB images
///        kornia::image::Image<float, 1> for grayscale float images
///
/// Supported types:
/// - Image<uint8_t, 1>  - Grayscale 8-bit
/// - Image<uint8_t, 3>  - RGB 8-bit
/// - Image<uint8_t, 4>  - RGBA 8-bit
/// - Image<float, 1>    - Grayscale 32-bit float
/// - Image<float, 3>    - RGB 32-bit float
/// - Image<float, 4>    - RGBA 32-bit float
template <typename T, size_t C> class Image {
    using Traits = ImageTraits<T, C>;
    using RustType = typename Traits::RustType_;

  public:
    explicit Image(rust::Box<RustType> img) : img_(std::move(img)) {
    }

    Image(size_t width, size_t height, T value) : img_(Traits::new_image(width, height, value)) {
    }

    Image(size_t width, size_t height, const std::vector<T>& data)
        : img_(Traits::from_data(width, height, rust::Slice<const T>(data.data(), data.size()))) {
    }

    /// Constructor from raw pointer (zero-copy via rust::Slice)
    /// @param width Image width in pixels
    /// @param height Image height in pixels
    /// @param data Raw pointer to pixel data (must be width * height * channels elements)
    /// @note The data is copied during construction. The caller retains ownership of the input data.
    /// @note explicit to avoid ambiguity with value constructor when passing 0/nullptr
    explicit Image(size_t width, size_t height, const T* data)
        : img_(Traits::from_data(width, height, rust::Slice<const T>(data, width * height * C))) {
    }

    Image(const Image&) = delete;
    Image& operator=(const Image&) = delete;
    Image(Image&&) = default;
    Image& operator=(Image&&) = default;

    size_t width() const {
        return Traits::width(*img_);
    }
    size_t height() const {
        return Traits::height(*img_);
    }
    size_t channels() const {
        return Traits::channels(*img_);
    }
    ImageSize size() const {
        return Traits::size(*img_);
    }
    rust::Slice<const T> data() const {
        return Traits::data(*img_);
    }
    std::vector<T> to_vec() const {
        auto slice = data();
        return std::vector<T>(slice.begin(), slice.end());
    }

    const RustType& inner() const {
        return *img_;
    }

  private:
    rust::Box<RustType> img_;
};

// Type aliases for convenience
#define KORNIA_IMAGE_TYPE(CppType, TypeName, FnPrefix, Channels)                                   \
    using TypeName = Image<CppType, Channels>;

KORNIA_IMAGE_TYPES
#undef KORNIA_IMAGE_TYPE

#undef KORNIA_IMAGE_TYPES

} // namespace image
} // namespace kornia
