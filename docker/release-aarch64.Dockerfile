FROM ghcr.io/rust-cross/manylinux2014-cross:aarch64

# rust image comes with sh, we like bash more
SHELL ["/bin/bash", "-c"]

RUN apt-get update --fix-missing && \
    apt-get install -y --no-install-recommends \
    gcc-multilib \
    && \
    apt-get clean
