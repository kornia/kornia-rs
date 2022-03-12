#!/bin/bash -ex

USERNAME=$(whoami)
USER_UID=$(id -u $USERNAME)
USER_GID=$(id -g $USERNAME)

docker build \
       --build-arg PARALLEL=$(nproc --ignore=2) \
       --build-arg USERNAME=$USERNAME \
       --build-arg USER_UID=$USER_UID \
       --build-arg USER_GID=$USER_GID \
       -f devel.Dockerfile \
       -t kornia_rs/devel:local ./
