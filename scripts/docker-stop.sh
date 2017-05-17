#!/usr/bin/env bash

# Mount the directory in the countainer and run in docker.
source scripts/init.sh
if [ ! "$(docker ps | grep $CONTAINER_NAME)" ]; then
    echo "That was an accident. Skipping..."
    #docker rm $CONTAINER_NAME
fi
