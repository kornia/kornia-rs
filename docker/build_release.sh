#!/bin/bash -ex

docker build \
       --build-arg PARALLEL=$(nproc --ignore=2) \
       -f release.Dockerfile \
       -t kornia_rs/release:local ./
