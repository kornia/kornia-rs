#pragma once

#include "kornia/version.hpp"

// Core Image types
#include "kornia/image.hpp"

// Core I/O functionality
#include "kornia/io.hpp"

namespace kornia {

inline const char* version() {
    return detail::get_version();
}

} // namespace kornia
