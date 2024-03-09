FROM rust:latest

RUN apt-get update --fix-missing && \
    apt-get install -y --no-install-recommends \
    cmake \
    nasm \
    && \
    apt-get clean
