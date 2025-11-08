cmake_minimum_required(VERSION 3.15)

# Fetch Catch2 v3 (modern, with main included)
include(FetchContent)
FetchContent_Declare(
    Catch2
    GIT_REPOSITORY https://github.com/catchorg/Catch2.git
    GIT_TAG v3.5.0  # Pin to specific stable version
    GIT_SHALLOW TRUE
)
FetchContent_MakeAvailable(Catch2)

# Test executable
add_executable(kornia_cpp_tests
    test_io_jpeg.cpp
)

target_link_libraries(kornia_cpp_tests
    PRIVATE
        kornia_cpp
        Catch2::Catch2WithMain
)

# Set C++ standard (match parent project)
target_compile_features(kornia_cpp_tests PRIVATE cxx_std_17)

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

