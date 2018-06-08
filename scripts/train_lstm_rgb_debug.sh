#!/bin/bash

set -e
experiment='LSTM_RGB_debug'

set -x
python main.py --experiment ${experiment} --mode train --num_workers 32 --max_example_per_label 100 --weight_decay 0.05 --use_cuda
