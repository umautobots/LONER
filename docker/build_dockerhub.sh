#!/usr/bin/env bash

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
IMAGE_TAG="loner:base_1.0"

DOCKER_OPTIONS=""
DOCKER_OPTIONS+="-t $IMAGE_TAG "
DOCKER_OPTIONS+="-f $SCRIPT_DIR/container_dockerhub.Dockerfile "

DOCKER_CMD="docker build $DOCKER_OPTIONS $SCRIPT_DIR"
echo $DOCKER_CMD
exec $DOCKER_CMD