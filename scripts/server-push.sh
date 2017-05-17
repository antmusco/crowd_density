#!/usr/bin/env bash

# Author: Anthony G. Musco
# Date: 05/01/2017
#
# Push files to the server from the local machine. This script requries you to
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

# Push source, config, and scripts.
rsync -av ccnn/    bigbrain:~/crowd_density/ccnn/
rsync -av config/  bigbrain:~/crowd_density/config/
rsync -av scripts/ bigbrain:~/crowd_density/scripts/

# Push image data (examples generated on the server).
rsync -av data/images/     bigbrain:~/crowd_density/data/images/
rsync -av data/image_sets/ bigbrain:~/crowd_density/data/image_sets/
