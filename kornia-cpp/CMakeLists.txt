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

# Create imported library for Rust (makes dependency explicit during build)
add_library(kornia_rust STATIC IMPORTED GLOBAL)
set_target_properties(kornia_rust PROPERTIES
    IMPORTED_LOCATION ${RUST_LIB_PATH}
)
add_dependencies(kornia_rust kornia_rust_lib)

# Create C++ library (header-only wrapper + Rust lib)
add_library(kornia_cpp INTERFACE)

# Set include directories (allows both #include <kornia.hpp> and #include <kornia/image.hpp>)
target_include_directories(kornia_cpp
    INTERFACE
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include/kornia>
        $<BUILD_INTERFACE:${CXXBRIDGE_DIR}>
        $<INSTALL_INTERFACE:include>
        $<INSTALL_INTERFACE:include/kornia>
)

# Link with Rust library
# In build tree: use kornia_rust IMPORTED target
# In install tree: reference kornia_rust imported target (created by config file)
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
    INCLUDES DESTINATION include
)

# Install headers
install(DIRECTORY include/
    DESTINATION include
    FILES_MATCHING PATTERN "*.hpp"
)

# Install Rust library
install(FILES ${RUST_LIB_PATH}
    DESTINATION lib
)

# Install CXX bridge headers (custom script to resolve symlinks)
install(CODE "
    file(GLOB_RECURSE CXX_HEADERS 
        \"${CXXBRIDGE_DIR}/*.h\" 
        \"${CXXBRIDGE_DIR}/*.hxx\"
    )
    foreach(HEADER \${CXX_HEADERS})
        file(RELATIVE_PATH REL_PATH \"${CXXBRIDGE_DIR}\" \"\${HEADER}\")
        get_filename_component(DEST_DIR \"\${CMAKE_INSTALL_PREFIX}/include/\${REL_PATH}\" DIRECTORY)
        file(INSTALL \"\${HEADER}\" DESTINATION \"\${DEST_DIR}\" FOLLOW_SYMLINK_CHAIN)
    endforeach()
")

# Install CMake targets file
install(EXPORT kornia-targets
    FILE kornia-targets.cmake
    NAMESPACE kornia::
    DESTINATION lib/cmake/kornia
)

# Generate and install package config files
include(CMakePackageConfigHelpers)

# Generate Rust imported target file (no variables to substitute, just copy)
configure_file(
    "${CMAKE_CURRENT_SOURCE_DIR}/cmake/kornia-rust-targets.cmake.in"
    "${CMAKE_CURRENT_BINARY_DIR}/kornia-rust-targets.cmake"
    COPYONLY
)

# Generate main package config file
configure_package_config_file(
    "${CMAKE_CURRENT_SOURCE_DIR}/cmake/kornia-config.cmake.in"
    "${CMAKE_CURRENT_BINARY_DIR}/kornia-config.cmake"
    INSTALL_DESTINATION lib/cmake/kornia
)

install(FILES
    "${CMAKE_CURRENT_BINARY_DIR}/kornia-config.cmake"
    "${CMAKE_CURRENT_BINARY_DIR}/kornia-rust-targets.cmake"
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

