# -*- coding: utf-8 -*-

import math
import time
import os
import shutil
import argparse

parser = argparse.ArgumentParser(description='LSTM FOR GEN DATA PROCESS')
parser.add_argument('--lr', type=float, default=1e-5)
parser.add_argument('--device', type=str, default='cuda')
parser.add_argument('--cuda_device_number', type=str, default='cuda')
parser.add_argument('--reload', type=bool, default=False)
parser.add_argument('--hidden_size', type=int, default=128)
parser.add_argument('--model_name', type=str, default='DeepQC')
parser.add_argument('--val_batch_size', type=int, default=10240)
parser.add_argument('--train_batch_size', type=int, default=10240)
parser.add_argument('--epoch', type=int, default=1000)
opt = parser.parse_args()

def time_since(since):
    s = time.time() - since
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)

def create_dir(path):
    if os.path.isdir(path):
        shutil.rmtree(path)
    os.makedirs(path)
