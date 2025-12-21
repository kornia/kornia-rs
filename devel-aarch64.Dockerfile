FROM ghcr.io/cross-rs/aarch64-unknown-linux-gnu:edge

RUN apt-get update && dpkg --add-architecture arm64 && apt-get update

RUN apt-get install --assume-yes \
    clang \
    cmake \
    nasm \
    protobuf-compiler \
    libgstreamer1.0-dev:arm64 \
    libgstreamer-plugins-base1.0-dev:arm64 \
    libssl-dev:arm64 \
    libglib2.0-dev:arm64 \
    && \
    apt-get clean
