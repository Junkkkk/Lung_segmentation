#!/bin/bash

python3 train.py --gpu 4 --model_name fcn32s &
python3 train.py --gpu 5 --model_name fcn16s &
python3 train.py --gpu 6 --model_name fcn8s &
python3 train.py --gpu 7 --model_name unet