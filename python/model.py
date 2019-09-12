# -*- coding: utf-8 -*-

import torch as t
import torch.nn as nn

class ResidualBlock(nn.Module):
    def __init__(self,din,dmol):
        super(ResidualBlock, self).__init__()
        self.con1=nn.Conv1d(in_channels=din,out_channels=dmol,kernel_size=1,bias=False)
        self.con2=nn.Conv1d(in_channels=dmol,out_channels=dmol,kernel_size=3,padding=1,bias=False)
        self.l=nn.Linear(3*dmol,dmol)
        self.short_cut=nn.Conv1d(in_channels=din,out_channels=dmol,kernel_size=1,bias=True)

    def forward(self, input,net):
        res=self.con1(input)
        res2=self.con2(t.cat((net.mem2,net.mem1,res),dim=-1))
        net.mem2=net.mem1
        net.mem1=res.detach()
        res2=self.l(res2.reshape(-1,45))
        output=self.short_cut(input).reshape(-1,15)+res2

        return output

class GenRNN(t.nn.Module):
    def __init__(self, input_size, hidden_size, output_size, seq_len=1):
        super(GenRNN, self).__init__()
        self.seq_len=seq_len
        self.input_size=input_size
        self.hidden_size=hidden_size
        self.output_size=output_size
        self.hidden_dim = hidden_size
        self.mem1=t.zeros((input_size,15,1)).cuda()
        self.mem2=t.zeros((input_size,15,1)).cuda()
        self.res1=ResidualBlock(1,15)
        self.mix_1=t.nn.LSTMCell(self.output_size+15,hidden_size)
        self.mix_2 = t.nn.LSTMCell(hidden_size+self.output_size+15, hidden_size)
        self.fc = t.nn.Linear(2*hidden_size, output_size)
        self.idx=0

    def forward(self, inputs, h,c,pos):
        if(self.idx%(self.seq_len-1)==0):
            self.mem1=t.zeros((self.input_size,15,1)).cuda()
            self.mem2=t.zeros((self.input_size,15,1)).cuda()

        seq_len,batch_size,feature = inputs.shape
        output=self.res1(inputs.transpose(0,1),self)
        output=t.cat((output,pos.expand(self.input_size,self.output_size)),dim=-1)
        h_0,c_0=self.mix_1(output,(h[0],c[0]))
        h_1,c_1=self.mix_2(t.cat((output,h_0),dim=-1),(h[1],c[1]))
        output = self.fc(t.cat((h_0.contiguous().view(batch_size, -1),h_1.contiguous().view(batch_size, -1)),dim=1))
        self.idx += 1

        return output,(h_0,h_1),(c_0,c_1),pos
