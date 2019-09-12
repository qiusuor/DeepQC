# -*- coding: utf-8 -*-
import sys

if len(sys.argv[1])<3:
    print("Usage: python extract_qual.py input.fastq output.qs")
    exit(-1)

outFile=open(sys.argv[2],'w')
with open(sys.argv[1]) as f:
    for i,x in enumerate(f):
        if i%4==3:
            outFile.write(x)

