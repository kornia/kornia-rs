FROM rust:latest

RUN apt-get update --fix-missing && \
    apt-get install -y --no-install-recommends \
    sudo \
    pkg-config \
    git \
    python3-pip \
    libssl-dev \
    libturbojpeg0-dev \
    libgtk-3-dev \
    && \
    apt-get clean

RUN python3 -m pip install --user --upgrade pip setuptools setuptools_rust

WORKDIR /workspace
