FROM quay.io/pypa/manylinux2010_x86_64

# rust image comes with sh, we like bash more
SHELL ["/bin/bash", "-c"]

RUN yum -y update && \
    yum -y install cmake \
                   clang \
                   python3 \
                   python-devel \
                   python3-devel \
                   python3-pip \
                   openssl-devel \
                   gtk3-devel \
                   yasm \
                   && yum -y clean all \
                   && rm -rf /var/cache

RUN curl --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --default-toolchain nightly
RUN source $HOME/.cargo/env

WORKDIR /workspace
