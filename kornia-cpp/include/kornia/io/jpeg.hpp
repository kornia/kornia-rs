#pragma once

#include "kornia/image.hpp"
#include <string>
#include <vector>

namespace kornia {
namespace io {

/// @brief Read an RGB JPEG image from file (u8, 3 channels)
///
/// @param file_path Path to JPEG file
/// @return ImageU8C3 containing RGB data
/// @throws rust::Error if file cannot be read or is invalid
///
/// @note Memory is managed by Rust. Zero-copy data access.
inline ImageU8C3 read_jpeg_rgb8(const std::string& file_path) {
    return ImageU8C3(::read_jpeg_rgb8(file_path)); // Call FFI function from global namespace
}

/// @brief Read a grayscale JPEG image from file (u8, 1 channel)
///
/// @param file_path Path to JPEG file
/// @return ImageU8C1 containing grayscale data
/// @throws rust::Error if file cannot be read or is invalid
///
/// @note Memory is managed by Rust. Zero-copy data access.
inline ImageU8C1 read_jpeg_mono8(const std::string& file_path) {
    return ImageU8C1(::read_jpeg_mono8(file_path)); // Call FFI function from global namespace
}

/// @brief Encode an RGB JPEG image to bytes (returns std::vector)
///
/// @param image RGB u8 image to encode
/// @param quality JPEG quality (0-100, where 100 is highest quality)
/// @return std::vector<uint8_t> containing JPEG-encoded bytes
/// @throws rust::Error if encoding fails
///
/// @note Copies data from rust::Vec to std::vector for C++ interop
inline std::vector<uint8_t> encode_jpeg_rgb8(const ImageU8C3& image, uint8_t quality) {
    auto bytes = ::encode_jpeg_rgb8(image.inner(), quality);
    return std::vector<uint8_t>(bytes.begin(), bytes.end());
}

/// @brief Decode JPEG bytes to RGB u8 image
///
/// @param jpeg_bytes JPEG-encoded bytes (borrowed via const reference)
/// @return ImageU8C3 containing decoded RGB data
/// @throws rust::Error if decoding fails or not RGB
///
/// @note Zero-copy: creates rust::Slice view of std::vector. Perfect for Unreal/OpenCV interop.
inline ImageU8C3 decode_jpeg_rgb8(const std::vector<uint8_t>& jpeg_bytes) {
    // Zero-copy: rust::Slice is just a view into the std::vector
    rust::Slice<const uint8_t> slice(jpeg_bytes.data(), jpeg_bytes.size());
    return ImageU8C3(::decode_jpeg_rgb8(slice));
}

} // namespace io
} // namespace kornia
