#!/bin/bash

if [ "$#" -ne 1 ]; then
    echo "Usage: $0 <mount-volume>"
    exit 1
fi

# Get the mount volume from the first argument
MOUNT_VOLUME=$1

CONTAINER_ID=$(
    docker run \
        -d \
        -e RUST_LOG=debug \
        -v $MOUNT_VOLUME:$MOUNT_VOLUME \
        -p 3000:3000 \
        kornia-serve
)

stop_docker() {
    docker stop $CONTAINER_ID
}

trap stop_docker SIGINT

docker logs -f $CONTAINER_ID &
LOG_PID=$!

docker wait $CONTAINER_ID

kill $LOG_PID
