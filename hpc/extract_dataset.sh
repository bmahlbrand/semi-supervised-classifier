#!/bin/bash

DIRECTORY=/scratch/$USER/data

if [ ! -d "$DIRECTORY" ]; then
  # Control will enter here if $DIRECTORY doesn't exist.
  tar -C /scratch/$USER -xvf /scratch/mg5439/ssl_data_96.tar.gz
  mv /scratch/$USER/ssl_data_96 $DIRECTORY
fi


