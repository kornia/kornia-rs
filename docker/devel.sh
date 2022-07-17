#!/bin/bash -e

KORNIA_IMAGE=${KORNIA_IMAGE:-"ghcr.io/kornia/kornia-rs/devel:latest"}

bash_args=$@
if [[ -z "$bash_args" ]] ; then
    bash_args="bash"
fi

test -t 1 && USE_TTY="-t"
set -x

docker run -i \
       $USE_TTY \
       -w /workspace \
       -v $(pwd):/workspace \
       $KORNIA_IMAGE \
       bash -c "$bash_args"
