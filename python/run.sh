#!/usr/bin/env bash

python train.py  --cuda_device_number '0' --epoch 1000 --val_batch_size 10240 --train_batch_size 10240 >log

