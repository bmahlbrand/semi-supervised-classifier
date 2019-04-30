#!/bin/bash

DIRECTORY=/scratch/$USER/checkpoints

if [ ! -d "$DIRECTORY" ]; then
  # Control will enter here if $DIRECTORY doesn't exist.
  mkdir $DIRECTORY
  ln -s $DIRECTORY ../checkpoints
fi


