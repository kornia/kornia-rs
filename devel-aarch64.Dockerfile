FROM ghcr.io/cross-rs/aarch64-unknown-linux-gnu:main

RUN dpkg --add-architecture arm64 && \
    apt-get update && \
    apt-get install --assume-yes \
    cmake \
    nasm:arm64 \
    && \
    apt-get clean
