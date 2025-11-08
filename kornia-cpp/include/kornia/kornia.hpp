#pragma once

#include "kornia/version.hpp"

// Core types
#include "kornia/image.hpp"

// I/O functionality
#include "kornia/io.hpp"

namespace kornia {

inline const char* version() {
    return detail::get_version();
}

} // namespace kornia
