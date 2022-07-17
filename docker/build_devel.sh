#!/bin/bash -ex

docker build \
       --build-arg PARALLEL=$(nproc --ignore=2) \
       -f devel.Dockerfile \
       -t kornia_rs/devel:local ./
