cmake_minimum_required(VERSION 3.15)
project(kornia_image_api_example VERSION 1.0 LANGUAGES CXX)

# Set C++ standard
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Assume we're using the local pre-built version
# kornia_cpp target should already exist from parent build
if(NOT TARGET kornia_cpp)
    # If building this example directly, add the parent library
    add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/../.. ${CMAKE_CURRENT_BINARY_DIR}/kornia)
endif()

# Create the example executable
add_executable(image_api_example main.cpp)

# Link with kornia_cpp (works both ways)
target_link_libraries(image_api_example PRIVATE kornia_cpp)

