#!/bin/bash

source loadmodules.sh
source activate deep
module list
# srun  --mem=8GB --gres=gpu:3 -c3  --pty /bin/bash
srun --time=5:00:00 --gres=gpu:2 -c2 --pty /bin/bash
