#!/usr/bin/env bash

# Author: Anthony G. Musco
# Date: 05/01/2017
#
# Run the test script on the server. This script requries the following:
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
# This script ssh's into the server and sets up the server environment to run
# the testing operation. First we change LD_LIBRARY_PATH to make sure
# TensorFlow uses CUDA 7.5 instead of 8.0 (be sure to change this if the server
# gets updated). Next, we activate the python virtual enviornment located at
# '~/tensorflow'. Finally, we change to the project directory and run the
# testing operation.

ssh -tt bigbrain << EOF
  set -e
  LD_LIBRARY_PATH=/usr/local/cuda-7.5/lib64:$LD_LIBRARY_PATH
  source ~/tensorflow/bin/activate
  cd ~/crowd_density
  ./ccnn/test.py
EOF
