#!/usr/bin/env bash

# Mount the directory in the countainer and run in docker.
source scripts/init.sh
# Container already running.
if [[ $(docker ps | grep $CONTAINER_NAME) ]]; then
    echo "Warning! Container already running. Attaching to running instance..."
    docker attach $CONTAINER_NAME 
# Container was stopped.
elif [[ $(docker ps -a | grep $CONTAINER_NAME) ]]; then
    echo "Restarting stopped container..."
    docker start -ai $CONTAINER_NAME 
# Container not yet created.
else
    echo "Starting new container..."
    docker run -it --name crowd-gym -v $CROWD_DENSITY_HOME/crowd-gym:/usr/local/gym/crowd-gym erlerobotics/gym-gazebo
fi
