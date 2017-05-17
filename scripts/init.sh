# Author: Anthony G. Musco
# Date: 05/01/2017
#
# Sets up ENV variables for Gazebo. This file should be 'source'd, not
# executed (otherwise the ENV variables will not be set in our shell.
CROWD_DENSITY_HOME=$(echo "$(cd "$(dirname "$1")"; pwd)")
CONTAINER_NAME="crowd-gym"

# Variables needed for scripts.
export GAZEBO_MODEL_PATH=$CROWD_DENSITY_HOME/gazebo/models
export GAZEBO_WORLD_PATH=$CROWD_DENSITY_HOME/gazebo/worlds
export GAZEBO_PLUGIN_PATH=$CROWD_DENSITY_HOME/gazebo/build
