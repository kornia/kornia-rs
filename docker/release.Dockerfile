FROM quay.io/pypa/manylinux2014_x86_64

# rust image comes with sh, we like bash more
SHELL ["/bin/bash", "-c"]

RUN yum -y update && \
    yum -y install clang \
                   gtk3-devel \
                   libjpeg-turbo-devel \
                   llvm-toolset-7 \
                   openssl-devel \
                   python3 \
                   python-devel \
                   python3-devel \
                   python3-pip \
                   && yum -y clean all \
                   && rm -rf /var/cache

# needed for clang >5.0
#RUN echo "source /opt/rh/llvm-toolset-7/enable" >> /etc/bashrc

RUN curl --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --default-toolchain nightly
RUN source $HOME/.cargo/env

RUN pip3 install --upgrade pip
RUN pip3 install tomli setuptools_rust
RUN pip3 install maturin[patchelf]

# Enable the SCL for all bash scripts.
ENV BASH_ENV=/opt/rh/llvm-toolset-7/enable \
    ENV=/opt/rh/llvm-toolset-7/enable \
    PROMPT_COMMAND=". /opt/rh/llvm-toolset-7/enable" \
    CC="clang" \
    CXX="clang++"

WORKDIR /workspace
