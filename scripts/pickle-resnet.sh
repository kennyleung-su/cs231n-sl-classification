#!/bin/bash

set -e
experiment='RESNET18(RGB)-1.0'
max_example=10

set -x
python main.py --experiment ${experiment} --max_example_per_label 2 --mode pickle