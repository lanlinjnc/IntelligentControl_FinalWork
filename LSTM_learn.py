#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2022/7/1 13:38
# @Author  : lanlin


import torch
from torch import nn
import numpy as np
import matplotlib.pyplot as plt
# from dataset import data_train, data_test

batch_size = 64
time_step = 1
epoch = 1000
input_size = 3
output_size = 1
mid_dim = 8
mid_layers = 1


class RegLSTM(nn.Module):
    def __init__(self, inp_dim, out_dim, mid_dim, mid_layers, batch):
        """
        :param inp_dim: 单个样本点的特征维度，这里为 3
        :param out_dim: 所需预测输出的维度，这里为 1
        :param mid_dim: 每一层中含有多少个LSTM结构，这里为 8
        :param mid_layers: 有多少含有LSTM的中间层，这里为 1
        :param batch: 自定义的 batch_size
        """
        super(RegLSTM, self).__init__()

        self.rnn = nn.LSTM(inp_dim, mid_dim, mid_layers, batch_first=batch)  # rnn
        self.reg = nn.Sequential(
            nn.Linear(mid_dim, mid_dim),
            nn.Tanh(),
            nn.Linear(mid_dim, out_dim),
        )  # regression

    def forward(self, x):
        y = self.rnn(x)[0]  # y, (h, c) = self.rnn(x)

        batch_size, seq_len, hid_dim = y.shape
        y = y.reshape(-1, hid_dim)
        y = self.reg(y)
        y = y.reshape(batch_size, seq_len, -1)
        return y


def load_data():
    seq_number = np.array(
        [112., 118., 132., 129., 121., 135., 148., 148., 136., 119., 104.,
         118., 115., 126., 141., 135., 125., 149., 170., 170., 158., 133.,
         114., 140., 145., 150., 178., 163., 172., 178., 199., 199., 184.,
         162., 146., 166., 171., 180., 193., 181., 183., 218., 230., 242.,
         209., 191., 172., 194., 196., 196., 236., 235., 229., 243., 264.,
         272., 237., 211., 180., 201., 204., 188., 235., 227., 234., 264.,
         302., 293., 259., 229., 203., 229., 242., 233., 267., 269., 270.,
         315., 364., 347., 312., 274., 237., 278., 284., 277., 317., 313.,
         318., 374., 413., 405., 355., 306., 271., 306., 315., 301., 356.,
         348., 355., 422., 465., 467., 404., 347., 305., 336., 340., 318.,
         362., 348., 363., 435., 491., 505., 404., 359., 310., 337., 360.,
         342., 406., 396., 420., 472., 548., 559., 463., 407., 362., 405.,
         417., 391., 419., 461., 472., 535., 622., 606., 508., 461., 390.,
         432.], dtype=np.float32
    )
    seq_number = seq_number[:, np.newaxis]
    seq_year = np.arange(12)
    seq_month = np.arange(12)
    seq_year_month = np.transpose(
        [
            np.repeat(seq_year, len(seq_month)),  # 复制到相应维度大小
            np.tile(seq_month, len(seq_year)),
        ]
    )
    seq = np.concatenate((seq_number, seq_year_month), axis=1)
    # seq为[seq_numbel, year, month]，维度为[144,3]
    seq = (seq - seq.mean(axis=0)) / seq.std(axis=0)
    return (seq)


if __name__=="__main__":
    data = load_data()
    print(data)
    train_size = int(len(data) * 0.75)

    # 生成连续样本，单个样本大小[time_step, input_size]，共[train_size - time_step + 1]个
    data_sample = np.zeros((train_size - time_step + 1, time_step, input_size))
    label_sample = np.zeros((train_size - time_step + 1, time_step, output_size))
    for i in range(train_size - time_step + 1):
        data_sample[i] = data[i:i + time_step, :]
        label_sample[i] = data[i + 1:i + 1 + time_step, 0:1:]  # 提取下一行第一列的数据

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = RegLSTM(input_size, output_size, mid_dim, mid_layers, True).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=1e-2)

    for i in range(epoch):
        # 每次输入一个batch数据
        for j in range(int((train_size - time_step + 1) / batch_size)):
            train_X = data_sample[j * batch_size:(j + 1) * batch_size, :, :]
            train_Y = label_sample[j * batch_size:(j + 1) * batch_size, :, :]
            var_x = torch.tensor(train_X, dtype=torch.float32, device=device)  # [32,1,3]
            var_y = torch.tensor(train_Y, dtype=torch.float32, device=device)  # [32,1,1]
            out = net(var_x)  # [32,1,1]
            loss = criterion(out, var_y)  # [1,]
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        # 每次不够一个batch的剩下数据
        train_X = data_sample[(j + 1) * batch_size:, :, :]
        train_Y = label_sample[(j + 1) * batch_size:, :, :]
        var_x = torch.tensor(train_X, dtype=torch.float32, device=device)
        var_y = torch.tensor(train_Y, dtype=torch.float32, device=device)
        out = net(var_x)
        loss = criterion(out, var_y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if i % 100 == 0:
            print('Epoch: {:4}, Loss: {:.5f}'.format(i, loss.item()))

    net = net.eval()
    test_X = data[train_size:, :]  # [36,3]
    test_Y = data[train_size + time_step:, 0:1:]  # [35,1]# 注意因为是预测，所以要 +time_step
    test_y = list()  # [35,1]

    for i in range(test_X.shape[0] - time_step):
        test_x = test_X[i:time_step + i, :].reshape(1, time_step, input_size)
        test_x = torch.tensor(test_x, dtype=torch.float32, device=device)
        tem = net(test_x).cpu().data.numpy()
        test_y.append(tem[0][-1])

    test_y = np.array(test_y).reshape((-1, 1))  # [35,1]
    diff = test_y - test_Y
    l1_loss = np.mean(np.abs(diff))
    l2_loss = np.mean(diff ** 2)
    print("L1:{:.3f}    L2:{:.3f}".format(l1_loss, l2_loss))
    plt.plot(test_y, 'r', label='pred')
    plt.plot(test_Y, 'b', label='real', alpha=0.3)

    a = np.array([1, 2, 3])
    b = np.array([2, 2, 4])

    a = a.reshape(-1, 1)
    b = b.reshape(-1, 1)

    np.corrcoef(a, b)

    a.reshape((-1))




