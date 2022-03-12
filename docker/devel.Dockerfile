FROM rust:latest

# rust image comes with sh, we like bash more
SHELL ["/bin/bash", "-c"]

ARG USERNAME=kornian
ARG USER_UID=1000
ARG USER_GID=$USER_UID

#RUN groupadd --gid $USER_GID $USERNAME
#RUN useradd --uid $USER_UID --gid $USER_GID -m $USERNAME

RUN apt-get update --fix-missing && \
    apt-get install -y --no-install-recommends \
    sudo \
    pkg-config \
    git \
    python3-pip \
    python3-venv \
    libssl-dev \
    libturbojpeg0-dev \
    libgtk-3-dev \
    && \
    apt-get clean

RUN pip3 install maturin[patchelf]

#USER $USERNAME

WORKDIR /workspace
