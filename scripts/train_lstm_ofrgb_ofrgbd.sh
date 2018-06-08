#!/bin/bash

set -e
experiment='LSTM_OFRGB_OFRGBD_final'

set -x
python main.py --experiment ${experiment} --mode train --num_workers 32 --use_cuda