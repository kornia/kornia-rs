FROM ghcr.io/rust-cross/manylinux2014-cross:aarch64

# rust image comes with sh, we like bash more
SHELL ["/bin/bash", "-c"]

RUN uname -a

RUN apt-get update --fix-missing && \
    apt-get install -y --no-install-recommends \
    gcc-aarch64-linux-gnu \
    && \
    apt-get clean
