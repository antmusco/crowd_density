#!/usr/bin/env bash

# Author: Anthony G. Musco
# Date: 05/01/2017
#
# This script builds the gazebo image generator project in 'gazebo' and runs
# the server to generate images. If you wish to inspect the image generation
# process in gazebo, you can run 'gzclient' in another terminal while this
# script is executing to observe the gazebo environment.

# Exit immediately if we fail to build
set -e 

# Set ENV variables.
source scripts/init.sh

# Build the project.
pushd gazebo/build
cmake ..
make
popd

# Run the server.
gzserver gazebo/worlds/Heisen.world --verbose

