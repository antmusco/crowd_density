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
# into the server and generates examples from whatever images are there.

ssh -tt bigbrain << EOF
  set -e
  source ~/tensorflow/bin/activate
  cd crowd_density
  ccnn/generate-examples.py
EOF
