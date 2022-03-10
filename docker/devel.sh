#!/bin/bash -e
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
       kornia_rs/devel \
       bash -c "$bash_args"
