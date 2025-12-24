ARG CROSS_BASE_IMAGE=ghcr.io/cross-rs/aarch64-unknown-linux-gnu:main
FROM $CROSS_BASE_IMAGE

# Clean and update package lists
RUN rm -rf /var/lib/apt/lists/* && \
    apt-get update && \
    dpkg --add-architecture arm64 && \
    apt-get update

RUN apt-get install --assume-yes --no-install-recommends \
    clang \
    cmake \
    nasm \
    protobuf-compiler \
    libgstreamer1.0-dev:arm64 \
    libgstreamer-plugins-base1.0-dev:arm64 \
    libssl-dev:arm64 \
    libglib2.0-dev:arm64 \
    && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Set environment variables for cross-compilation pkg-config
ENV PKG_CONFIG_ALLOW_CROSS=1
ENV PKG_CONFIG_PATH=/usr/lib/aarch64-linux-gnu/pkgconfig
ENV PKG_CONFIG_SYSROOT_DIR=/
