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
    COMMAND ${CARGO_CMD}
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
    COMMENT "Building Kornia C++ Rust Crate..."
    BYPRODUCTS "${RUST_TARGET_DIR}/${RUST_LIB_NAME}"
)

# Create detail directory structure for cleaner includes during build
# Use CMake script to conditionally copy files (avoids shell quoting issues)
set(COPY_HEADERS_SCRIPT "${CMAKE_CURRENT_BINARY_DIR}/copy_cxx_headers.cmake")
file(WRITE ${COPY_HEADERS_SCRIPT}
"# Auto-generated script to copy CXX bridge headers
file(MAKE_DIRECTORY \"${CMAKE_CURRENT_SOURCE_DIR}/target/cxxbridge/kornia/detail\")

# Define expected header locations
set(LIB_RS_H_SRC \"${CMAKE_CURRENT_SOURCE_DIR}/target/cxxbridge/kornia-cpp/src/lib.rs.h\")
set(CXX_H_SRC \"${CMAKE_CURRENT_SOURCE_DIR}/target/cxxbridge/rust/cxx.h\")
set(LIB_RS_H_DST \"${CMAKE_CURRENT_SOURCE_DIR}/target/cxxbridge/kornia/detail/lib.rs.h\")
set(CXX_H_DST \"${CMAKE_CURRENT_SOURCE_DIR}/target/cxxbridge/kornia/detail/cxx.h\")

# Copy lib.rs.h if found
if(EXISTS \"${LIB_RS_H_SRC}\")
  configure_file(
    \"${LIB_RS_H_SRC}\"
    \"${LIB_RS_H_DST}\"
    COPYONLY
  )
  message(STATUS \"Copied lib.rs.h to kornia/detail/\")
else()
  message(FATAL_ERROR \"CXX bridge header not found: ${LIB_RS_H_SRC}\\nCXX headers must be generated during Rust build. Check that build.rs is running correctly.\")
endif()

# Copy cxx.h if found
if(EXISTS \"${CXX_H_SRC}\")
  configure_file(
    \"${CXX_H_SRC}\"
    \"${CXX_H_DST}\"
    COPYONLY
  )
  message(STATUS \"Copied cxx.h to kornia/detail/\")
else()
  message(FATAL_ERROR \"CXX header not found: ${CXX_H_SRC}\\nCXX headers must be generated during Rust build. Check that build.rs is running correctly.\")
endif()
")

# --- 3. Define Header Locations ---
set(CXX_GEN_DIR "${CMAKE_CURRENT_SOURCE_DIR}/target/cxxbridge")

# IMPORTANT: CXX generates the folder structure using the crate name (kornia-cpp with dash).
set(MAIN_HEADER "${CXX_GEN_DIR}/kornia-cpp/src/lib.rs.h")
set(RUST_CXX_HEADER "${CXX_GEN_DIR}/rust/cxx.h")

# Output header files (organized into kornia/detail/)
set(ORGANIZED_LIB_RS_H "${CXX_GEN_DIR}/kornia/detail/lib.rs.h")
set(ORGANIZED_CXX_H "${CXX_GEN_DIR}/kornia/detail/cxx.h")

# Create a command to organize headers - runs after cargo_build but before compilation
# Use OUTPUT so CMake knows these files must exist before targets that depend on them
# Note: We only depend on cargo_build, not the source headers, because they may not exist
# and CMake will try to build them as targets if we list them as DEPENDS
add_custom_command(
    OUTPUT ${ORGANIZED_LIB_RS_H} ${ORGANIZED_CXX_H}
    COMMAND ${CMAKE_COMMAND} -P ${COPY_HEADERS_SCRIPT}
    COMMENT "Organizing CXX bridge headers into kornia/detail/"
    DEPENDS cargo_build
    VERBATIM
)

# Create a target that produces the organized headers
add_custom_target(organize_cxx_headers ALL
    DEPENDS ${ORGANIZED_LIB_RS_H} ${ORGANIZED_CXX_H}
)

# Also run it as POST_BUILD for cargo_build to ensure it happens
add_custom_command(TARGET cargo_build POST_BUILD
    COMMAND ${CMAKE_COMMAND} -P ${COPY_HEADERS_SCRIPT}
    COMMENT "Organizing CXX bridge headers into kornia/detail/"
)

# Create imported target for the Rust static library
# The library file is created by cargo_build, so we mark it as a dependency
add_library(kornia_rust_lib STATIC IMPORTED GLOBAL)
set_target_properties(kornia_rust_lib PROPERTIES
    IMPORTED_LOCATION "${RUST_TARGET_DIR}/${RUST_LIB_NAME}"
)
# Ensure cargo_build completes before this imported library is used
# This tells CMake the file will be created by cargo_build
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

# Ensure headers are organized before any target uses kornia_cpp
# Note: add_dependencies on INTERFACE libraries doesn't work as expected,
# so we'll make individual targets depend on organize_cxx_headers directly

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
