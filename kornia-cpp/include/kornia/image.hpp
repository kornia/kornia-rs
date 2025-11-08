#pragma once

#include "kornia-cpp/src/lib.rs.h"

namespace kornia {

/// @brief Image container - thin wrapper around Rust ImageResult
///
/// This is a zero-copy type alias to the Rust ImageResult structure.
/// Data is stored in row-major order with interleaved channels.
/// Memory is managed by Rust.
using Image = ::ImageResult;

} // namespace kornia
