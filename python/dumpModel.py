# -*- coding: utf-8 -*-
import torch as t
import torch
from utils import opt
from model_cpp import GenRNN
from collections import Counter

batch_size = 10240
all_chars=Counter()
line=None
with open('data/train.qs') as f:
    for i,x in enumerate(f):
        if i>100000:
            break
        x=x.replace('\n','')
        line=x
        all_chars.update(x)
characters = sorted(all_chars.keys())
len_line=len(line)
int2char = dict(enumerate(characters))
char2int = {char: index for index, char in int2char.items()}

if __name__ == '__main__':
    model = GenRNN(input_size=10240, hidden_size=opt.hidden_size, output_size=len(characters),seq_len=len_line)
    device = t.device(opt.device)
    print(model)
    model.load_state_dict(t.load('./checkpoints/net_{}.pth'.format(opt.model_name),map_location='cuda:0'))
    model = model.to(device)
    model.eval()
    example = torch.rand(1,10240,1).to(device)
    h_0 = example.data.new(2, 10240, 128).fill_(0).float().to(device)
    c_0 = example.data.new(2, 10240, 128).fill_(0).float().to(device)
    h_state = (h_0, c_0)
    pos=t.zeros((10240,1)).cuda()
    mem_0,mem_1=(t.zeros(10240,15,1).cuda(),t.zeros(10240,15,1).cuda())
    traced_script_module = torch.jit.trace(model,(example,h_0,c_0,pos,mem_0,mem_1))
    traced_script_module.save("model.pt")
