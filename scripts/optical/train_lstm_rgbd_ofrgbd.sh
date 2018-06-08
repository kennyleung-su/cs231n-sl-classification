#!/bin/bash

set -e
experiment='LSTM_RGBD_OFRGBD_final'

set -x
python main.py --experiment ${experiment} --use_cuda --mode train --num_workers 32 --save_every_epoch --validate_every 1 --num_sweeps 5
