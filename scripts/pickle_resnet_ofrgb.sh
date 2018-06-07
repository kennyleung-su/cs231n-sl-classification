#!/bin/bash
# Pickle Optical Flow RGB ResNet encodings.

set -e
experiment='RN18_OFRGB_final'
checkpoint='RN18_OFRGB_checkpoint.pkl'

set -x
python main.py --mode pickle --experiment ${experiment} --use_cuda --num_workers 32 --load --checkpoint_to_load ${checkpoint}
