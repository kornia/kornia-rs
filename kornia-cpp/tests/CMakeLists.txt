cmake_minimum_required(VERSION 3.15)

# Use pre-installed Catch2 from pixi/system package manager
# This avoids network-dependent FetchContent at build time
find_package(Catch2 3 QUIET)

# Fallback to FetchContent if Catch2 is not found (for developer builds)
if(NOT Catch2_FOUND)
    include(FetchContent)
    FetchContent_Declare(
        Catch2
        GIT_REPOSITORY https://github.com/catchorg/Catch2.git
        GIT_TAG v3.5.0  # Pin to specific stable version
        GIT_SHALLOW TRUE
    )
    FetchContent_MakeAvailable(Catch2)
endif()

# Test executable
add_executable(kornia_cpp_tests
    test_image_buffer.cpp
    test_io_jpeg.cpp
    test_version.cpp
    test_image.cpp
)

target_link_libraries(kornia_cpp_tests
    PRIVATE
        kornia_cpp
        Catch2::Catch2WithMain
)

# C++ standard is inherited from parent project (CMAKE_CXX_STANDARD 17)

# Enable warnings
target_compile_options(kornia_cpp_tests PRIVATE
    $<$<CXX_COMPILER_ID:GNU,Clang>:-Wall -Wextra -Wpedantic>
    $<$<CXX_COMPILER_ID:MSVC>:/W4>
)

# Copy test data to build directory for easy access
file(COPY ${CMAKE_CURRENT_SOURCE_DIR}/../../tests/data
     DESTINATION ${CMAKE_CURRENT_BINARY_DIR})

# Enable CTest
enable_testing()

# Catch2 CTest integration - auto-discovers test cases
list(APPEND CMAKE_MODULE_PATH ${catch2_SOURCE_DIR}/extras)
include(CTest)
include(Catch)
catch_discover_tests(kornia_cpp_tests
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    TEST_PREFIX "kornia::"
)

# Optional: Add sanitizers for debug builds
if(CMAKE_BUILD_TYPE STREQUAL "Debug" AND NOT MSVC)
    option(ENABLE_SANITIZERS "Enable sanitizers (ASAN, UBSAN)" OFF)
    if(ENABLE_SANITIZERS)
        target_compile_options(kornia_cpp_tests PRIVATE
            -fsanitize=address,undefined
            -fno-omit-frame-pointer
        )
        target_link_options(kornia_cpp_tests PRIVATE
            -fsanitize=address,undefined
        )
    endif()
endif()
