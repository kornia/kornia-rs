FROM ghcr.io/cross-rs/aarch64-unknown-linux-gnu:main

RUN apt-get update && dpkg --add-architecture arm64 && apt-get update

RUN apt-get install --assume-yes \
    cmake \
    nasm \
    libgstreamer1.0-dev:arm64 \
    libgstreamer-plugins-base1.0-dev:arm64 \
    && \
    apt-get clean
