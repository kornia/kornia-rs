FROM quay.io/pypa/manylinux2014_x86_64

# rust image comes with sh, we like bash more
SHELL ["/bin/bash", "-c"]

# needed for clang >5.0
RUN yum -y update && \
    yum -y install llvm-toolset-7

RUN scl enable llvm-toolset-7 bash

# install other dependencies
RUN yum -y update && \
    yum -y install clang \
                   gtk3-devel \
                   libjpeg-turbo-devel \
                   openssl-devel \
                   python3 \
                   python-devel \
                   python3-devel \
                   python3-pip \
                   && yum -y clean all \
                   && rm -rf /var/cache

RUN curl --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --default-toolchain nightly
RUN source $HOME/.cargo/env

RUN pip3 install --upgrade pip
RUN pip3 install tomli setuptools_rust
RUN pip3 install maturin[patchelf]

WORKDIR /workspace
