FROM ghcr.io/cross-rs/i686-unknown-linux-gnu:main

RUN apt-get update && \
    apt-get install --assume-yes \
    cmake \
    gcc \
    nasm \
    libgstreamer1.0-dev \
    libgstreamer-plugins-base1.0-dev \
    && \
    apt-get clean
