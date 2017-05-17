#!/usr/bin/env bash

# Author: Anthony G. Musco
# Date: 05/01/2017
#
# Start the Gazebo client on the local machine, attaching it to the Gazebo
# server running in a docker container.

# Note: For this script to work, only a single docker container can be active.

# Get the id of the running docker container.
PID=`docker ps -q`

# Set the Gazebo IP to the IP of this container.
export GAZEBO_MASTER_IP=$(sudo docker inspect --format '{{ .NetworkSettings.IPAddress }}' "$PID")
export GAZEBO_MASTER_URI=$GAZEBO_MASTER_IP:11345

# Run the client.
gzclient
