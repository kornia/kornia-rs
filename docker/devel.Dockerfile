FROM rust:latest

# rust image comes with sh, we like bash more
SHELL ["/bin/bash", "-c"]

RUN apt-get update --fix-missing && \
    apt-get install -y --no-install-recommends \
    sudo \
    pkg-config \
    ca-certificates \
    build-essential \
    git \
    python3-dev \
    python3-pip \
    python3-venv \
    libssl-dev \
    libturbojpeg0-dev \
    libgtk-3-dev \
    && \
    apt-get clean

RUN pip3 install maturin[patchelf]

WORKDIR /workspace
