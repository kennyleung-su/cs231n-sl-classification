#!/bin/bash
# Pickle RGB ResNet encodings.

set -e
experiment='RN18_RGB_final'

set -x
python main.py --mode pickle --experiment ${experiment} --use_cuda --num_workers 32