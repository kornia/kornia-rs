FROM rust:1.85

RUN rustup update stable

RUN apt-get update --fix-missing && \
    apt-get install -y --no-install-recommends \
    cmake \
    nasm \
    libgstreamer1.0-dev \
    libgstreamer-plugins-base1.0-dev \
    && \
    apt-get clean
