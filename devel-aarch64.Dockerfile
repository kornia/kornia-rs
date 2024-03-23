FROM ghcr.io/cross-rs/aarch64-unknown-linux-gnu:main

RUN apt-get update && \
    apt-get install --assume-yes \
    cmake \
    nasm \
    libgstreamer1.0-dev \
    libgstreamer-plugins-base1.0-dev \
    && \
    apt-get clean
