#!/bin/bash

set -e
experiment='RESNET18(RGBD)-1.0'

set -x
python main.py --experiment ${experiment} --mode pickle
