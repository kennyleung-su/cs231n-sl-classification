#!/bin/bash
# Pickle RGB-D ResNet encodings.

set -e
experiment='RN18_RGBD_final'
checkpoint='RN18_RGBD_checkpoint.pkl'

set -x
python main.py --mode pickle --experiment ${experiment} --use_cuda --num_workers 32 --load --checkpoint_to_load ${checkpoint}
