FROM rust:latest

# rust image comes with sh, we like bash more
SHELL ["/bin/bash", "-c"]

RUN apt-get update --fix-missing && \
    apt-get install -y --no-install-recommends \
    build-essential \
    ca-certificates \
    cmake \
    git \
    libclang-dev \
    libssl-dev \
    libturbojpeg0-dev \
    libgtk-3-dev \
    nasm \
    pkg-config \
    python3-dev \
    python3-pip \
    python3-venv \
    sudo \
    && \
    apt-get clean

RUN pip3 install maturin[patchelf]
RUN pip3 install pre-commit

# add rust tools
RUN rustup component add rustfmt
RUN rustup component add clippy

WORKDIR /workspace
