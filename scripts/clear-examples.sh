#!/usr/bin/env bash

# Author: Anthony G. Musco
# Date: 05/01/2017
#
# This script deletes all examples from 'data/examples', as well as the example
# set lists from 'data/example_sets'.

# MAKE SURE USER WANTS TO DELETE DATA
echo "Are you sure you want to clear all examples?"
echo "This cannot be undone. [y/n]: "
read yn

case "$yn" in 
  [yY])
    echo "Okay, clearing examples..."
    rm -r data/examples/*
    rm -r data/example_sets/*
    ;;
esac
