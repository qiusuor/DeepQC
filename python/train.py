# -*- coding: utf-8 -*-

import os
import time
import torch as t
import numpy as np
import torchnet as tnt
from model import GenRNN
from torch.utils import data
from utils import time_since, create_dir,opt
os.environ['CUDA_VISIBLE_DEVICES'] = opt.cuda_device_number
from collections import Counter

all_chars=Counter()
line=None
chars_list=[Counter() for _ in range(1000)]
with open('data/train.qs') as f:
    for i,x in enumerate(f):
        x=x.replace('\n','')
        line=x
        all_chars.update(x)
        for i,c in enumerate(x):
            chars_list[i].update(c)

for i in range(len(line)):
    for k in all_chars.keys():
        if k not in chars_list[i].keys():
            chars_list[i][k]=1

n_num_lines=sum(chars_list[0].values())
positional_dis=[]
for i in range(len(line)):
    positional_dis.append([chars_list[i][x]/n_num_lines for x in sorted(all_chars.keys())])

positional_dis=t.tensor(positional_dis)

characters = sorted(all_chars.keys())
len_line=len(line)

int2char = dict(enumerate(characters))
char2int = {char: index for index, char in int2char.items()}
min_bpc=8.0


print(opt)
def data_process(file_name):
    with open(file_name) as tmp_f:
        line_len = len(tmp_f.readline())
    print("Line_len", line_len)
    text = open(file_name, 'r').read()
    text = text.replace('\n', '')
    encoded_text = [char2int[char] for char in text]
    np.savez_compressed(file_name.replace('.qs', '.npz'), data=np.array(encoded_text).reshape(-1, len_line))
    print("Data write to npz file")

class Dataset(data.Dataset):
    def __init__(self,train=True,batch_size=2048):
        self.index=0
        self.batch_size=batch_size
        self.batch_index=0
        if train:

            file_name = './data/train.npz'
            if not os.path.exists(file_name):
                data_process('./data/train.qs')
            self.data = np.load(file_name)['data']
        else:
            file_name = './data/test.npz'
            if not os.path.exists(file_name):
                data_process('./data/test.qs')
            self.data = np.load(file_name)['data']
        self.batch_num=(self.data.shape[0]) //self.batch_size

    def __getitem__(self, item):
        if self.batch_index >=self.batch_num:
            self.index=0
            self.batch_index=0
            raise StopIteration
        x,y= self.data[self.batch_index*self.batch_size:self.batch_index*self.batch_size+self.batch_size,self.index],self.data[self.batch_index*self.batch_size:self.batch_index*self.batch_size+self.batch_size,self.index+1]
        self.index += 1
        self.index %= (len_line-1)
        if self.index == 0:
            self.batch_index += 1
        return x,y


def val(model, val_data, device):
    model.eval()
    start_time = time.time()
    acc_over_steps=[[] for _ in range(len_line)]
    test_loss_meter = tnt.meter.AverageValueMeter()
    test_accuracy_meter = tnt.meter.AverageValueMeter()
    criterion = t.nn.CrossEntropyLoss().to(device)
    sum_all = 0
    time_num = 0
    print("Total batch for test: {}".format(val_data.batch_num*(len_line-1)))
    h, c = None, None
    for j,(x, y) in enumerate(val_data):
        if j % (len_line-1) == 0:
            h, c = t.zeros((opt.num_layers, opt.val_batch_size, opt.hidden_size)), t.zeros(
                (opt.num_layers, opt.val_batch_size, opt.hidden_size))
            h, c = h.to(device), c.to(device)
        x = x.reshape((1, opt.val_batch_size, 1))
        x = t.from_numpy(x).type(t.FloatTensor).to(device)
        y = t.from_numpy(y).type(t.LongTensor).to(device)
        output, h, c,_ = model(x, h, c,positional_dis[j%(len_line-1)].to(device))
        h, c = (h[0].detach(), h[1].detach()), (c[0].detach(), c[1].detach())
        loss = criterion(output, y.contiguous().view(-1))
        test_loss_meter.add(loss.item())
        output = t.nn.functional.softmax(output, dim=1)
        output_numpy = output.cpu().detach().numpy().astype(np.float32)
        target_numpy = y.contiguous().view(-1).cpu().detach().numpy()

        for i in range(target_numpy.shape[0]):
            sum_all += np.log2(output_numpy[i][target_numpy[i]])
            time_num += 1
        pred = np.argmax(output_numpy, axis=1)
        accuracy = float((pred == target_numpy).astype(int).sum()) / float(target_numpy.size)
        test_accuracy_meter.add(accuracy)
        acc_over_steps[j%(len_line-1)].append(accuracy)
        if j % (len_line-1) == (len_line-2):
            print("Step: {}, Time {}, Acc: {}".format(j, time_since(start_time), test_accuracy_meter.value()[0]))
    bpc = -1 * (sum_all / time_num)
    print("TestLoss: {}, TestAccuracy: {}, TestBpc: {}, TestTime: {}. TotalChars: {}".
          format(test_loss_meter.value()[0], test_accuracy_meter.value()[0], bpc, time_since(start_time), time_num))
    model.train()
    global min_bpc
    if(bpc<min_bpc):
        t.save(model.state_dict(), './checkpoints/net_{}.pth'.format(opt.model_name))
        min_bpc=bpc

def train():
    train_data = Dataset(train=True, batch_size=opt.train_batch_size)

    val_data = Dataset(train=False, batch_size=opt.val_batch_size)
    model = GenRNN(input_size=opt.train_batch_size, hidden_size=opt.hidden_size, output_size=len(characters), seq_len=len_line)
    device = t.device(opt.device)
    if opt.reload:
        print("model reload")
        model.load_state_dict(t.load('./checkpoints/net_{}.pth'.format(opt.model_name)))

    model = model.to(device)
    optimizer = t.optim.Adam(model.parameters(), lr=opt.lr)
    criterion = t.nn.CrossEntropyLoss().to(device)
    train_loss_meter = tnt.meter.AverageValueMeter()
    accuracy_meter = tnt.meter.AverageValueMeter()
    start_time = time.time()
    h,c=None,None
    for epoch in range(opt.epoch):
        step = 0
        for i,(x, y) in enumerate(train_data):
            if i%(len_line-1)==0:
                h,c = t.zeros((opt.num_layers,opt.train_batch_size,opt.hidden_size)),t.zeros((opt.num_layers,opt.train_batch_size,opt.hidden_size))
                h,c=h.to(device),c.to(device)
            step += 1
            model.zero_grad()
            x=x.reshape((1,opt.train_batch_size,1))
            x=t.from_numpy(x).type(t.FloatTensor).to(device)
            y=t.from_numpy(y).type(t.LongTensor).to(device)
            output, h, c,_ = model(x, h, c,positional_dis[i%(len_line-1)].to(device))
            h,c=(h[0].detach(),h[1].detach()),(c[0].detach(),c[1].detach())
            loss = criterion(output, y.contiguous().view(-1))
            train_loss_meter.add(loss.item())
            loss.backward()
            optimizer.step()
            output = t.nn.functional.softmax(output, dim=1)
            output_numpy = output.cpu().detach().numpy()
            target_numpy = y.contiguous().view(-1).cpu().detach().numpy()
            pred = np.argmax(output_numpy, axis=1)
            accuracy = float((pred == target_numpy).astype(int).sum()) / float(target_numpy.size)
            accuracy_meter.add(accuracy)
            if step % ((len_line-1)*200) > 0 and step %((len_line-1)*200) <= (len_line-1):
                sum_all = 0
                time_num = 0
                for i in range(target_numpy.shape[0]):
                    sum_all += np.log2(output_numpy[i][target_numpy[i]])
                    time_num += 1
                bpc = -1 * (sum_all / time_num)
                print('Epoch: {}, Step: {}, Loss: {}, accuracy: {}, Time: {}, TrainBpc: {}, CompRatio: {}'
                      .format(epoch, (step-1) % (len_line-1), train_loss_meter.value()[0], accuracy, time_since(start_time), bpc, bpc / 8))

                train_loss_meter.reset()
                accuracy_meter.reset()
        val(model=model, val_data=val_data, device=device)

if __name__ == '__main__':
    create_dir('./logs/{}'.format(opt.model_name))
    train()

