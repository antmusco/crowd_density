#!/usr/bin/env bash

# Author: Anthony G. Musco
# Date: 05/01/2017
#
# This script deletes all images from 'data/images', as well as the image set
# lists from 'data/image_sets'.

# MAKE SURE USER WANTS TO DELETE DATA
echo "Are you sure you want to clear all images?"
echo "This cannot be undone. [y/n]: "
read yn

case "$yn" in 
  [yY])
    echo "Okay, clearing images..."
    rm -r data/images/*
    rm -r data/image_sets/*
    ;;
esac
