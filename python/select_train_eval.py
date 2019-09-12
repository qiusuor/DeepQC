# -*- coding: utf-8 -*-

import sys
import random
import numpy as np

if len(sys.argv[1])<2:
    print("Usage: python select_train_eval.py dataSet.qs")
    exit(-1)

selected=[]
with open(sys.argv[1]) as f:
    for i,x in enumerate(f):
        if i%4==3:
            if random.randint(0,10)==0:
                selected.append(x)


np.random.shuffle(selected)

train_file=open("data/train.qs",'w')
train_file.writelines(selected[0:len(selected)//2])
test_file=open("data/test.qs",'w')
test_file.writelines(selected[len(selected)//2:])


