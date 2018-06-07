#!/bin/bash

set -e
experiment='LSTM_RGB_RGBD_final'

set -x
python main.py --experiment ${experiment} --use_cuda --mode train --num_workers 32