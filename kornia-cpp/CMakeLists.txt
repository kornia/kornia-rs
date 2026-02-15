cmake_minimum_required(VERSION 3.12)

project(KorniaCppLib VERSION 1.0 LANGUAGES CXX)

# C++ Standard
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# --- 1. Settings ---
if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    set(RUST_TARGET_DIR "${CMAKE_CURRENT_SOURCE_DIR}/target/debug")
    set(CARGO_CMD cargo build)
else()
    set(RUST_TARGET_DIR "${CMAKE_CURRENT_SOURCE_DIR}/target/release")
    set(CARGO_CMD cargo build --release)
endif()

# Use static library (required for CXX bridge C++ symbol export)
if(WIN32)
    set(RUST_LIB_NAME "kornia_cpp.lib")
else()
    set(RUST_LIB_NAME "libkornia_cpp.a")
endif()

# --- 2. Build Rust ---
add_custom_target(cargo_build ALL
    COMMAND ${CMAKE_COMMAND} -E env CARGO_TARGET_DIR=${CMAKE_CURRENT_SOURCE_DIR}/target ${CARGO_CMD}
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
    COMMENT "Building Kornia C++ Rust Crate..."
    BYPRODUCTS "${RUST_TARGET_DIR}/${RUST_LIB_NAME}"
)

# Create detail directory structure for cleaner includes during build
add_custom_command(TARGET cargo_build POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E make_directory ${CMAKE_CURRENT_SOURCE_DIR}/target/cxxbridge/kornia/detail
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
        ${CMAKE_CURRENT_SOURCE_DIR}/target/cxxbridge/kornia-cpp/src/lib.rs.h
        ${CMAKE_CURRENT_SOURCE_DIR}/target/cxxbridge/kornia/detail/lib.rs.h
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
        ${CMAKE_CURRENT_SOURCE_DIR}/target/cxxbridge/rust/cxx.h
        ${CMAKE_CURRENT_SOURCE_DIR}/target/cxxbridge/kornia/detail/cxx.h
    COMMENT "Organizing CXX bridge headers into kornia/detail/"
)

# --- 3. Define Header Locations ---
set(CXX_GEN_DIR "${CMAKE_CURRENT_SOURCE_DIR}/target/cxxbridge")

# IMPORTANT: CXX generates the folder structure using the crate name (kornia-cpp with dash).
set(MAIN_HEADER "${CXX_GEN_DIR}/kornia-cpp/src/lib.rs.h")
set(RUST_CXX_HEADER "${CXX_GEN_DIR}/rust/cxx.h")

# Create imported target for the Rust static library
add_library(kornia_rust_lib STATIC IMPORTED GLOBAL)
set_target_properties(kornia_rust_lib PROPERTIES
    IMPORTED_LOCATION "${RUST_TARGET_DIR}/${RUST_LIB_NAME}"
)
add_dependencies(kornia_rust_lib cargo_build)

# Create kornia_cpp as INTERFACE library that wraps everything
add_library(kornia_cpp INTERFACE)
target_include_directories(kornia_cpp INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include/kornia>
    $<BUILD_INTERFACE:${CXX_GEN_DIR}>
    $<INSTALL_INTERFACE:include>
    $<INSTALL_INTERFACE:include/kornia>
)

# Link the Rust library
target_link_libraries(kornia_cpp INTERFACE kornia_rust_lib)

# --- 4. Link System Libraries ---
# Note: We don't explicitly link stdc++ or libc++ here because:
# - The consuming project (like Unreal Engine) provides its own C++ stdlib
# - Unreal uses bundled libc++, explicitly linking stdc++ causes conflicts
# - The C++ stdlib will be automatically linked by the final executable
if(APPLE)
    # macOS: dl functions are part of the system, only pthread is needed
    target_link_libraries(kornia_cpp INTERFACE pthread)
elseif(UNIX)
    # Linux: needs both dl and pthread
    target_link_libraries(kornia_cpp INTERFACE dl pthread)
elseif(WIN32)
    # Windows handles C++ stdlib automatically, but may need these:
    target_link_libraries(kornia_cpp INTERFACE ws2_32 userenv bcrypt ntdll)
endif()

# --- 5. Install Rules ---
include(GNUInstallDirs)

# Install libkornia_cpp.a
install(FILES "${RUST_TARGET_DIR}/${RUST_LIB_NAME}"
    DESTINATION ${CMAKE_INSTALL_LIBDIR}
)

# Install C++ wrapper headers from include/
install(DIRECTORY include/
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
    FILES_MATCHING PATTERN "*.hpp"
)

# Install CXX bridge headers (into kornia/detail/ for cleaner structure)
install(CODE "
    file(GLOB_RECURSE CXX_HEADERS
        \"${CXX_GEN_DIR}/*.h\"
        \"${CXX_GEN_DIR}/*.hxx\"
    )
    foreach(HEADER \${CXX_HEADERS})
        get_filename_component(HEADER_NAME \"\${HEADER}\" NAME)
        # Install all CXX bridge headers into kornia/detail/ directory
        file(INSTALL \"\${HEADER}\"
             DESTINATION \"\${CMAKE_INSTALL_PREFIX}/${CMAKE_INSTALL_INCLUDEDIR}/kornia/detail\"
             FOLLOW_SYMLINK_CHAIN)
    endforeach()
")

# --- 6. Examples ---
option(BUILD_EXAMPLES "Build examples" ON)
if(BUILD_EXAMPLES)
    add_subdirectory(examples/read_jpeg)
    add_subdirectory(examples/image_api)
endif()

# --- 7. Tests ---
option(BUILD_TESTS "Build tests" OFF)
if(BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()
