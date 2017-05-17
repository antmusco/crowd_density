#!/usr/bin/env bash

# Author: Anthony G. Musco
# Date: 05/01/2017
#
# Pull files from the server to the local machine. This script requries you to
# create a 'bigbrain' host in ~/.ssh/config, as well as a symlink to the NFS
# director containing the project in your home directory:
#
# ```
#   Host bigbrain
#     HostName bigbrain.cs.stonybrook.edu
#     Port 130
#     User user_name
#     ForwardX11Trusted yes
# ```
# 
# ```
#   ln -s /nfs/bigbrain/user_name/crowd_density ~/crowd_density
# ```

# Pull the logs and checkpoints
rsync -av bigbrain:~/crowd_density/logs/train/checkpoint logs/train/
rsync -av bigbrain:~/crowd_density/logs/test/hist.png logs/test/
