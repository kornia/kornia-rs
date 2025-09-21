FROM rust:1

RUN rustup update stable
RUN rustup component add clippy
ENV OPENCV_PKGCONFIG_NAME=opencv4

RUN apt-get update --fix-missing && \
    apt-get install -y --no-install-recommends \
    clang \
    cmake \
    ninja-build \
    pkg-config \
    nasm \
    libopencv-dev \
    libturbojpeg-dev \
    libv4l-dev \
    libgstreamer1.0-dev \
    libgstreamer-plugins-base1.0-dev \
    && \
    apt-get clean
