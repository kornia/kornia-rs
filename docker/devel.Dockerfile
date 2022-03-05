FROM rust:slim-buster

RUN apt-get update --fix-missing && \
    apt-get install -y --no-install-recommends \
    sudo \
    pkg-config \
    git \
    python3-pip \
    libssl-dev \
    libturbojpeg0-dev \
    libegl1-mesa-dev \
    && \
    apt-get clean

RUN python3 -m pip install --upgrade pip setuptools setuptools_rust
