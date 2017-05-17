#!/usr/bin/env bash

# Author: Anthony G. Musco
# Date: 05/01/2017
#
# Run TensorBoard on the server, and forward it's port to our LOCAL machine.
# This script requries the following:
#
# 1. A 'bigbrain' host in ~/.ssh/config on your LOCAL machine:
#   ```
#   # In ~/.ssh/config
#   Host bigbrain
#     HostName bigbrain.cs.stonybrook.edu
#     Port 130
#     User user_name
#     ForwardX11Trusted yes
#   ```
# 2. A symlink to the project directory (located on the server NFS) in your
#    SERVER home directory. 
#    ```
#    # In bigbrain:~/
#      ln -s /nfs/bigbrain/user_name/crowd_density ~/crowd_density
#    ```
# 3. A python virtual environment in your SERVER home directory:
#    # In bigbrain:~/
#    virtualenv ~/tensorflow
#    source ~/tensorflow/bin/activate
#    pip install -r ~/crowd_density/tensorflow-requirements.txt
#
# This script should be run from your local machine, not the server. It ssh's
# into the server, forwarding server port 6006 to local port 6006. It then runs
# tensorboard, reading from the logdir. If everything works right, you should
# now be able to view TensorBoard in your browser at 127.0.0.1:6006
#
# Note: We kill all running instances of tensorboard that may have 

# Make sure we kill the tensorboard process on the server if we get interrupted
trap ctrl_c INT
function ctrl_c() {
  echo "Killing TensorBoard process on server... "
  ssh bigbrain << EOF
killall -9 tensorboard 
EOF
}

ssh -L 6006:130.245.4.240:6006 bigbrain << EOF
killall -9 tensorboard
source ~/tensorflow/bin/activate
tensorboard --logdir ~/crowd_density/logs/train
EOF
