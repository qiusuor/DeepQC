#!/usr/bin/env bash

rm data/* -rf

ln -s $1 data/input.fastq

python extract_qual.py data/input.fastq data/input.qs

python select_train_eval.py data/input.qs

python train.py --epoch 1000 --val_batch_size 10240 --train_batch_size 10240

python dumpModel.py

python get_stat_info.py data/test.qs


