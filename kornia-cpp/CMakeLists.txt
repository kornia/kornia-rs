cmake_minimum_required(VERSION 3.15)
project(kornia-cpp VERSION 0.1.0 LANGUAGES CXX)

# Set C++ standard
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Build Rust library first
set(CARGO_TARGET_DIR ${CMAKE_CURRENT_SOURCE_DIR}/target)

# Cargo uses lowercase profile names
if(CMAKE_BUILD_TYPE STREQUAL "Release")
    set(CARGO_PROFILE "release")
    set(RUST_LIB_DIR ${CARGO_TARGET_DIR}/release)
else()
    set(CARGO_PROFILE "debug")
    set(RUST_LIB_DIR ${CARGO_TARGET_DIR}/debug)
endif()

# Determine the library name based on platform
if(WIN32)
    set(RUST_LIB_NAME "kornia_cpp.lib")
    set(RUST_LIB_PATH "${RUST_LIB_DIR}/${RUST_LIB_NAME}")
else()
    set(RUST_LIB_NAME "libkornia_cpp.a")
    set(RUST_LIB_PATH "${RUST_LIB_DIR}/${RUST_LIB_NAME}")
endif()

# Build the Rust library as a custom command (runs first)
add_custom_command(
    OUTPUT ${RUST_LIB_PATH}
    COMMAND cargo build --color=always $<$<CONFIG:Release>:--release>
    DEPENDS 
        ${CMAKE_CURRENT_SOURCE_DIR}/Cargo.toml
        ${CMAKE_CURRENT_SOURCE_DIR}/src/lib.rs
        ${CMAKE_CURRENT_SOURCE_DIR}/build.rs
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
    COMMENT "Building Rust library with Cargo (${CARGO_PROFILE} profile)"
    USES_TERMINAL
)

# Create a custom target that depends on the Rust library
add_custom_target(kornia_rust_lib
    DEPENDS ${RUST_LIB_PATH}
)

# Find the CXX bridge generated headers
set(CXXBRIDGE_DIR ${CARGO_TARGET_DIR}/cxxbridge)

# Create imported library for Rust (makes dependency explicit)
add_library(kornia_rust STATIC IMPORTED GLOBAL)
set_target_properties(kornia_rust PROPERTIES
    IMPORTED_LOCATION ${RUST_LIB_PATH}
)
add_dependencies(kornia_rust kornia_rust_lib)

# Create C++ library (header-only wrapper + Rust lib)
add_library(kornia_cpp INTERFACE)

# Set include directories (header-only)
target_include_directories(kornia_cpp
    INTERFACE
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<BUILD_INTERFACE:${CXXBRIDGE_DIR}>
        $<INSTALL_INTERFACE:include>
)

# Link with Rust library (now properly imported)
target_link_libraries(kornia_cpp
    INTERFACE
        kornia_rust
)

# Handle platform-specific system libraries
if(UNIX AND NOT APPLE)
    target_link_libraries(kornia_cpp INTERFACE pthread dl m)
elseif(APPLE)
    target_link_libraries(kornia_cpp INTERFACE pthread)
endif()

# Installation rules
install(TARGETS kornia_cpp
    EXPORT kornia-targets
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
    RUNTIME DESTINATION bin
    INCLUDES DESTINATION include
)

install(DIRECTORY include/
    DESTINATION include
    FILES_MATCHING PATTERN "*.hpp"
)

install(EXPORT kornia-targets
    FILE kornia-targets.cmake
    NAMESPACE kornia::
    DESTINATION lib/cmake/kornia
)

# Create config file
include(CMakePackageConfigHelpers)
write_basic_package_version_file(
    "${CMAKE_CURRENT_BINARY_DIR}/kornia-config-version.cmake"
    VERSION ${PROJECT_VERSION}
    COMPATIBILITY AnyNewerVersion
)

configure_package_config_file(
    "${CMAKE_CURRENT_SOURCE_DIR}/cmake/kornia-config.cmake.in"
    "${CMAKE_CURRENT_BINARY_DIR}/kornia-config.cmake"
    INSTALL_DESTINATION lib/cmake/kornia
)

install(FILES
    "${CMAKE_CURRENT_BINARY_DIR}/kornia-config.cmake"
    "${CMAKE_CURRENT_BINARY_DIR}/kornia-config-version.cmake"
    DESTINATION lib/cmake/kornia
)

# Examples (optional)
option(BUILD_EXAMPLES "Build examples" ON)
if(BUILD_EXAMPLES)
    add_subdirectory(examples/read_jpeg)
endif()

# Tests (optional)
option(BUILD_TESTS "Build tests" OFF)
if(BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()

