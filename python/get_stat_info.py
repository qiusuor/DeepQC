# -*- coding: utf-8 -*-

import sys
from collections import Counter
chars_list=[Counter() for _ in range(10000)]
chars=Counter()
all_chars=Counter()
line=None
with open(sys.argv[1]) as f:
    for i,x in enumerate(f):
        if i>1000000:
            break
        x=x.replace('\n','')
        line=x
        all_chars.update(x)
        chars.update(x[0])
        for i,c in enumerate(x):
            chars_list[i].update(c)

for i in range(len(line)):
    for k in all_chars.keys():
        if k not in chars_list[i].keys():
            chars_list[i][k]=1

with open(sys.argv[1]) as f:
    for x in f:
        assert (len(x)==len(line)+1)

chars_sorted=sorted(chars.keys())
cumFreq=list()
all_chars_sorted=sorted(all_chars.keys())

for i in range(len(all_chars)):
    if all_chars_sorted[i] in chars.keys():
        cumFreq.append(chars[all_chars_sorted[i]])
    else:
        cumFreq.append(1)
for i in range(1, len(all_chars_sorted)):
    cumFreq[i]+=cumFreq[i-1]

all_chars_sorted=[str(x) for x in all_chars_sorted]
cumFreq=[str(x) for x in cumFreq]
with open('stat_info','w') as f:
    f.write(str(len(all_chars.keys())) + '\n')
    f.write('\t'.join([str(x) for x in sorted(all_chars.keys())]) + '\n')
    f.write(str(len(line)) + '\n')
    f.write(str(len(all_chars.keys()))+'\n')
    f.write('\t'.join(all_chars_sorted)+'\n')
    f.write('\t'.join(cumFreq)+'\n')
    f.write(str(sum(chars_list[0].values()))+'\n')
    for i in range(len(line)):
        f.write('\t'.join([str(chars_list[i][x]) for x in sorted(all_chars.keys())]) + '\n')

