#!/bin/bash
# TODO(yinghang)

set -e
experiment='LSTM_combined_final'

set -x
python main.py --experiment ${experiment} --use_cuda --mode train --num_workers 32