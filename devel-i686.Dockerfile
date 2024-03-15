FROM ghcr.io/cross-rs/i686-unknown-linux-gnu:main

RUN apt-get update && \
    apt-get install --assume-yes \
    cmake \
    gcc \
    nasm \
    && \
    apt-get clean
