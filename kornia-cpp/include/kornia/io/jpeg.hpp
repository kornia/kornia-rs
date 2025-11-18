#pragma once

#include "kornia/image.hpp"
#include <string>
#include <vector>

namespace kornia {
namespace io {
namespace jpeg {

// Import image types into io namespace for convenience
using kornia::image::ImageBuffer;
using kornia::image::ImageU8C1;
using kornia::image::ImageU8C3;

/// @brief Read an RGB JPEG image from file (u8, 3 channels)
///
/// Uses libjpeg-turbo for high-performance JPEG decoding.
/// Image data is owned by Rust and accessed via zero-copy rust::Slice.
///
/// @note Memory is managed by Rust. Zero-copy data access.
inline ImageU8C3 read_jpeg_rgb8(const std::string& file_path) {
    return ImageU8C3(::read_image_jpeg_rgb8(file_path));
}

/// @brief Read a grayscale JPEG image from file (u8, 1 channel)
///
/// Uses libjpeg-turbo for high-performance JPEG decoding.
/// If source is color, it will be converted to grayscale.
/// Image data is owned by Rust and accessed via zero-copy rust::Slice.
///
/// @note Memory is managed by Rust. Zero-copy data access.
inline ImageU8C1 read_jpeg_mono8(const std::string& file_path) {
    return ImageU8C1(::read_image_jpeg_mono8(file_path));
}

/// @brief Encode an RGB JPEG image to bytes (zero-copy, into ImageBuffer)
///
/// @param image RGB u8 image to encode
/// @param quality JPEG quality (0-100, where 100 is highest quality)
/// @param buffer ImageBuffer to write JPEG data into (zero-copy)
/// @throws rust::Error if encoding fails
///
/// @note RECOMMENDED for performance. No copies - data stays in Rust-managed memory.
///       Access via buffer.data()/size() or buffer.as_span() for zero-copy operations.
///       The same buffer can be reused for different formats (JPEG, PNG, etc.).
///
/// @example
/// @code
/// kornia::io::ImageBuffer buffer;
/// for (const auto& image : images) {
///     buffer.clear();
///     kornia::io::encode_jpeg_rgb8(image, 95, buffer);
///     send_network(buffer.data(), buffer.size());  // Zero-copy
/// }
/// @endcode
inline void encode_image_jpeg_rgb8(const ImageU8C3& image, uint8_t quality, ImageBuffer& buffer) {
    ::encode_image_jpeg_rgb8(image.inner(), quality, buffer.rust_vec());
}

/// @brief Decode JPEG bytes to RGB u8 image
///
/// @param jpeg_bytes JPEG-encoded bytes (borrowed via const reference)
/// @return ImageU8C3 containing decoded RGB data
/// @throws rust::Error if decoding fails or not RGB
///
/// @note Zero-copy: creates rust::Slice view of std::vector. Perfect for Unreal/OpenCV interop.
inline ImageU8C3 decode_image_jpeg_rgb8(const std::vector<uint8_t>& jpeg_bytes) {
    // Zero-copy: rust::Slice is just a view into the std::vector
    rust::Slice<const uint8_t> slice(jpeg_bytes.data(), jpeg_bytes.size());
    return ImageU8C3(::decode_image_jpeg_rgb8(slice));
}

} // namespace jpeg
} // namespace io
} // namespace kornia
