#!/bin/bash

set -e
experiment='LSTM_RGB_OFRGB_debug'

set -x
python main.py --experiment ${experiment} --mode train --num_workers 32 --use_cuda
