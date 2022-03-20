FROM quay.io/pypa/manylinux2010_x86_64

# rust image comes with sh, we like bash more
SHELL ["/bin/bash", "-c"]

ARG USERNAME=kornian
ARG USER_UID=1000
ARG USER_GID=$USER_UID

#RUN groupadd --gid $USER_GID $USERNAME
#RUN useradd --uid $USER_UID --gid $USER_GID -m $USERNAME

RUN uname -a

RUN yum -y update && \
    yum -y install python3 \
                   python-devel \
                   python3-devel \
                   python3-pip \
                   openssl-devel \
                   libjpeg-turbo-devel \
                   gtk3-devel \
                   && yum -y clean all \
                   && rm -rf /var/cache

RUN curl --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --default-toolchain nightly
RUN source $HOME/.cargo/env

WORKDIR /workspace