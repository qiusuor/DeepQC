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

    def forward(self, input,mem_0,mem_1):
        res=self.con1(input)
        res2=self.con2(t.cat((mem_1,mem_0,res),dim=-1))
        mem_1 = mem_0
        mem_0=res.detach()
        res2=self.l(res2.reshape(-1,45))
        output=self.short_cut(input).reshape(-1,15)+res2

        return output,mem_0,mem_1

class GenRNN(t.nn.Module):
    def __init__(self, input_size, hidden_size, output_size, seq_len=1):
        super(GenRNN, self).__init__()
        self.seq_len=seq_len
        self.output_size=output_size
        self.input_size=input_size
        self.hidden_size=hidden_size
        self.hidden_dim = hidden_size
        self.res1=ResidualBlock(1,15)
        self.mix_1 = t.nn.LSTMCell(self.output_size + 15, hidden_size)
        self.mix_2 = t.nn.LSTMCell(hidden_size + self.output_size + 15, hidden_size)
        self.fc = t.nn.Linear(2*hidden_size, output_size)

    def forward(self, inputs, h,c,pos,mem_0,mem_1):
        _,batch_size,feature = inputs.shape
        output,mem_0,mem_1=self.res1(inputs.transpose(0,1),mem_0,mem_1)
        output = t.cat((output, pos.expand(self.input_size, self.output_size)), dim=-1)
        h_0,c_0=self.mix_1(output,(h[0],c[0]))
        h_1,c_1=self.mix_2(t.cat((output,h_0),dim=-1),(h[1],c[1]))
        tc = t.cat((h_0.contiguous().reshape((batch_size, -1)),h_1.contiguous().reshape((batch_size, -1))),dim=1)
        output = self.fc(tc)
        ts=t.nn.functional.softmax(output, dim=1)
        tff=t.floor(ts * 1e8)
        output = tff.int()
        output[output == 0] = 1
        for i in range(output.shape[1] - 1):
            output[:, i + 1] += output[:, i]
        return output.reshape((-1)), t.stack((h_0,h_1),dim=0),t.stack((c_0,c_1),dim=0),mem_0,mem_1

