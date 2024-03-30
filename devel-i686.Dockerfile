FROM ghcr.io/cross-rs/i686-unknown-linux-gnu:main

RUN apt-get update && dpkg --add-architecture i386 && apt-get update

RUN apt-get install --assume-yes \
    cmake \
    gcc \
    nasm \
    libgstreamer1.0-dev:i386 \
    libgstreamer-plugins-base1.0-dev:i386 \
    && \
    apt-get clean
