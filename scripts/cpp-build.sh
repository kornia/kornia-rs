#!/bin/bash
# Helper script for C++ builds
# Usage: cpp-build.sh [options]
#   --release       Build in Release mode (default: Debug)
#   --tests         Enable tests
#   --examples      Enable examples
#   --sanitizers    Enable sanitizers
#   --run-tests     Run tests after building
#   --install       Install after building

set -e

BUILD_TYPE="Debug"
BUILD_TESTS="OFF"
BUILD_EXAMPLES="OFF"
ENABLE_SANITIZERS="OFF"
RUN_TESTS=false
INSTALL=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --release) BUILD_TYPE="Release"; shift ;;
        --tests) BUILD_TESTS="ON"; shift ;;
        --examples) BUILD_EXAMPLES="ON"; shift ;;
        --sanitizers) ENABLE_SANITIZERS="ON"; shift ;;
        --run-tests) RUN_TESTS=true; BUILD_TESTS="ON"; shift ;;
        --install) INSTALL=true; shift ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

# Detect number of cores
NPROC=$(nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 4)

echo "Building C++ with:"
echo "  BUILD_TYPE=$BUILD_TYPE"
echo "  BUILD_TESTS=$BUILD_TESTS"
echo "  BUILD_EXAMPLES=$BUILD_EXAMPLES"
echo "  ENABLE_SANITIZERS=$ENABLE_SANITIZERS"
echo "  NPROC=$NPROC"

# Configure
cmake -B build \
    -DCMAKE_BUILD_TYPE="$BUILD_TYPE" \
    -DBUILD_TESTS="$BUILD_TESTS" \
    -DBUILD_EXAMPLES="$BUILD_EXAMPLES" \
    -DENABLE_SANITIZERS="$ENABLE_SANITIZERS"

# Build
if [ "$BUILD_TESTS" = "OFF" ] && [ "$BUILD_EXAMPLES" = "OFF" ]; then
    cmake --build build --target cargo_build -j"$NPROC"
else
    cmake --build build -j"$NPROC"
fi

# Run tests if requested
if [ "$RUN_TESTS" = true ]; then
    cd build && ctest --output-on-failure
    cd ..
fi

# Install if requested
if [ "$INSTALL" = true ]; then
    sudo cmake --install build
fi
