FROM ghcr.io/cross-rs/aarch64-unknown-linux-gnu:main

RUN apt-get update && \
    apt-get install --assume-yes \
    cmake \
    nasm \
    && \
    apt-get clean
