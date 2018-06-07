#!/bin/bash
# Pickle Optical Flow RGB-D ResNet encodings.

set -e
experiment='RN18_OFRGBD_final'
checkpoint='RN18_OFRGBD_checkpoint.pkl'

set -x
python main.py --mode pickle --experiment ${experiment} --use_cuda --num_workers 32 --load --checkpoint_to_load ${checkpoint}
