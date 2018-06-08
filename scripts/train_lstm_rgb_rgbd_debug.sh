#!/bin/bash

set -e
experiment='LSTM_RGB_RGBD_debug'

set -x
python main.py --experiment ${experiment} --mode train --num_workers 0