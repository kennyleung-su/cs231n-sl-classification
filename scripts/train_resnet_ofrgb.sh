#!/bin/bash
# Training Optical Flow RGB ResNet encodings.

set -e
experiment='RN18_OFRGB_final'

set -x
python main.py --mode train --experiment ${experiment} --use_cuda --num_workers 32 --save_every_epoch