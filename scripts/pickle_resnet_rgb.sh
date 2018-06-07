#!/bin/bash
# Pickle RGB ResNet encodings.

set -e
experiment='RN18_RGB_final'

set -x
python main.py --mode pickle --experiment ${experiment} --num_workers 0 --max_example_per_label 2