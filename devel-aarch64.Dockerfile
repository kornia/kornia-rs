FROM ghcr.io/cross-rs/aarch64-unknown-linux-gnu:edge

RUN apt-get update && dpkg --add-architecture arm64 && apt-get update

ENV OPENCV_PKGCONFIG_NAME=opencv4

RUN apt-get install --assume-yes \
    clang \
    cmake \
    ninja-build \
    pkg-config \
    nasm \
    libclang-dev:arm64 \
    libopencv-dev:arm64 \
    libturbojpeg-dev:arm64 \
    libv4l-dev:arm64 \
    libgstreamer1.0-dev:arm64 \
    libgstreamer-plugins-base1.0-dev:arm64 \
    libssl-dev:arm64 \
    libglib2.0-dev:arm64 \
    && \
    apt-get clean
