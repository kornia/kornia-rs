#!/bin/bash

# Initialize the data directory to an empty string
DATA_DIR=""

# Process the options
while (( "$#" )); do
  case "$1" in
    --data-dir)
      if [ -n "$2" ] && [ ${2:0:1} != "-" ]; then
        DATA_DIR=$2
        shift 2
      else
        echo "Error: Argument for $1 is missing" >&2
        exit 1
      fi
      ;;
    -*|--*=) # unsupported flags
      echo "Error: Unsupported flag $1" >&2
      exit 1
      ;;
    *) # preserve positional arguments
      PARAMS="$PARAMS $1"
      shift
      ;;
  esac
done

# Check if the data directory is provided
if [ -z "$DATA_DIR" ]; then
    echo "Usage: $0 --data-dir <data-directory>"
    exit 1
fi

CONTAINER_ID=$(
    docker run \
        -it \
        -d \
        -e RUST_LOG=debug \
        -v $DATA_DIR:$DATA_DIR \
        -p 3000:3000 \
        kornia-serve
)

stop_docker() {
    echo "Stopping container $CONTAINER_ID"
    trap '' SIGINT
    docker stop $CONTAINER_ID > /dev/null
    echo "Container stopped successfully. Goodbyte!"
    exit 0
}

trap stop_docker SIGINT

docker logs -f $CONTAINER_ID &
LOG_PID=$!

docker wait $CONTAINER_ID

kill $LOG_PID
