# -*- coding: utf-8 -*-

import sys
if len(sys.argv[1])<3:
    print("Usage: python discard_qual.py input.qs output.qs")
    exit(-1)

out_f=open(sys.argv[2],'w')
with open(sys.argv[1]) as f:
    for i,x in enumerate(f):
        if i%4==3:
            out_f.write('T'*(len(x)-1)+'\n')
        else:
            out_f.write(x)

