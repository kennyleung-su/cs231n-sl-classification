#!/bin/bash

set -e
experiment='LSTM_RGB_debug'

set -x
python main.py --experiment ${experiment} --mode train --num_workers 0 --max_example_per_label 2 --verbose