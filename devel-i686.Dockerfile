FROM ghcr.io/cross-rs/i686-unknown-linux-gnu:main

RUN apt-get update && dpkg --add-architecture i386 && apt-get update

ENV OPENCV_PKGCONFIG_NAME=opencv4

RUN apt-get install --assume-yes \
    cmake \
    ninja-build \
    pkg-config \
    gcc \
    nasm \
    libclang-dev:i386 \
    libopencv-dev:i386 \
    libturbojpeg-dev:i386 \
    libv4l-dev:i386 \
    libgstreamer1.0-dev:i386 \
    libgstreamer-plugins-base1.0-dev:i386 \
    && \
    apt-get clean
