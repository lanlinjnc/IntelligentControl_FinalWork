#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2022/7/9 22:21
# @Author  : lanlin
# 思路2：根据u(t),I(t)预测y(t)，故输入数据维度为2或者3（3是指时序位置）


import torch
from torch import nn
import numpy as np
import matplotlib.pyplot as plt
from dataset import dataset


batch_size = 2000
time_step = 4  # 不大于400，否则报错
epoch = 2000
input_size = 3
output_size = 1
mid_dim = 4
mid_layers = 2


# 随机生成2000训练集和400测试集
data_index = [i for i in range(dataset.shape[0]-time_step)]
used_data_index = np.random.choice(data_index, 2400, replace=False)  # 抽取坐标
used_test_index = np.random.choice(used_data_index, 400, replace=False)
used_train_index = [i for i in used_data_index if i not in used_test_index]
used_train_index = np.array(used_train_index)
unused_data_index = [i for i in data_index if i not in used_data_index]
# for i in used_test_index:
#     if i in used_train_index:
#         print("error")


class Mydataset(torch.utils.data.Dataset):

    def __init__(self, data_array, order_array, transform=None, target_transform=None):
        """
        :param data_array: 输入的完整数据数组
        :param order_array: 输入的data_array中的所需次序
        :param transform:
        :param target_transform:
        """
        self.datas = data_array
        self.orders = order_array
        self.transform = transform
        self.target_transform = target_transform

    def __getitem__(self, index):  # 只需设计第index个数据
        input_data = np.zeros((time_step, input_size), dtype=np.float32)
        y_data = np.zeros((time_step, output_size), dtype=np.float32)
        for i in range(time_step):
            input_data[i] = self.datas[index+i][0:3]
            y_data[i] = self.datas[index + i][-1]
        if self.transform:
            input_data = self.transform(input_data)
        return input_data, y_data  # 返回单个样本对（数据、标签）

    def __len__(self):
        return len(self.orders)


class MyLSTM(nn.Module):
    def __init__(self, inp_dim, out_dim, mid_dim, mid_layers, batch=True):
        """
        :param inp_dim: 单个样本点的特征维度，这里为 3
        :param out_dim: 所需预测输出的维度，这里为 1
        :param mid_dim: 每一层中含有多少个LSTM结构，这里为 8
        :param mid_layers: 有多少含有LSTM的中间层，这里为 1
        :param batch: 自定义的 batch_size
        """
        super(MyLSTM, self).__init__()

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


def LSTM_mean(data_2D):

    temp_2D = np.zeros((data_2D.shape[0], data_2D.shape[0]+3), dtype=np.float32)
    for i in range(temp_2D.shape[0]):
        temp_2D[i][i:i+4] = data_2D[i][:]
    # data_1D = np.mean(temp_2D, 0)  # 沿着0方向取均值
    exist = (temp_2D != 0)
    num = np.sum(temp_2D, axis=0)
    den = exist.sum(axis=0)
    data_1D = num/den
    return data_1D


if __name__=="__main__":

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = MyLSTM(input_size, output_size, mid_dim, mid_layers).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=1e-2)

    """ ---------------------使用Mydataset类进行训练和测试-----------------------"""
    data_train = Mydataset(dataset, used_train_index)  # 训练集初始化
    train_loader = torch.utils.data.DataLoader(data_train, batch_size=batch_size)

    data_test = Mydataset(dataset, used_test_index)  # 测试集初始化
    test_loader = torch.utils.data.DataLoader(data_test, batch_size=400)

    data_test_rest = Mydataset(dataset, unused_data_index)  # 原始数据中剩余的数据集进行测试
    test_rest_loader = torch.utils.data.DataLoader(data_test_rest, batch_size=450)
    # 开始训练，并进行测试
    train_loss_record = []  # 保存训练数据
    test_loss_record = []  # 保存测试数据
    for i in range(epoch):
        net.train()
        loss_train_steps = 0.0
        for j, (train_X, train_Y) in enumerate(train_loader):
            # print(j)
            train_X = train_X.to(device)
            train_Y = train_Y.to(device)
            out_train = net(train_X)  # [32,1,1]
            loss_train = criterion(out_train, train_Y)  # [1,]
            optimizer.zero_grad()
            loss_train.backward()
            optimizer.step()
            loss_train_steps += loss_train.item()  # 保存loss
        train_loss_record.append(loss_train_steps/(j+1))


        # 开始测试
        net = net.eval()
        for j, (test_X, test_Y) in enumerate(test_loader):
            test_X = test_X.to(device)
            test_Y = test_Y.to(device)
            out_test = net(test_X)  # [32,1,1]
            loss_test = criterion(out_test, test_Y)  # [1,]
            test_loss_record.append(loss_test.item())  # 保存loss

        # 打印训练参数
        if i % 100 == 0:
            print('Epoch: {:4}, train Loss: {:.5f}, Test Loss: {:.5f}'
                  .format(i, loss_train_steps/(j+1), loss_test.item()))

    torch.save(net.state_dict(), './weights/LSTM_params.pth')
    #
    # # 测试剩余的数据
    # # net_paras_path = './weights/LSTM_params.pth'
    # # state_dict = torch.load(net_paras_path)
    # # net.load_state_dict(state_dict)
    # # net = net.eval()
    # # # rest_loss_record = []  # 保存测试数据
    # # for j, (test_X, test_Y) in enumerate(test_rest_loader):
    # #     test_X = test_X.to(device)
    # #     test_Y = test_Y.to(device)
    # #     out_test = net(test_X)
    # #     loss_test = criterion(out_test, test_Y)
    # #     print(loss_test.item())  # 保存loss
    # # out_test = out_test.cpu().squeeze(2).detach().numpy()
    # # test_Y = test_Y.cpu().squeeze(2).detach().numpy()

    """ ------------------------绘制损失函数和精度----------------------------"""
    fig_name = "LSTMNet"
    fontsize = 15
    fig, (ax1, ax2) = plt.subplots(2, figsize=(15, 12), sharex=True)
    ax1.plot(train_loss_record)
    ax1.set_ylabel("train loss", fontsize=fontsize)
    ax1.set_title(fig_name, fontsize="xx-large")
    ax2.plot(test_loss_record)
    ax2.set_ylabel("test loss", fontsize=fontsize)
    ax2.set_xlabel("epochs", fontsize=fontsize)
    plt.tight_layout()
    plt.savefig('./results/' + fig_name + '.png')
    plt.show()

    """ ------------------------利用所有数据与原始函数y对比----------------------------"""
    data_test_rest = Mydataset(dataset, unused_data_index)  # 原始数据中剩余的数据集进行测试
    test_rest_loader = torch.utils.data.DataLoader(data_test_rest, batch_size=450)

    # read LSTM model
    data_test = Mydataset(dataset, data_index)
    all_loader = torch.utils.data.DataLoader(data_test, batch_size=3000)

    model = MyLSTM(input_size, output_size, mid_dim, mid_layers)
    model_paras_path = './weights/LSTM_params.pth'
    state_dict = torch.load(model_paras_path)
    model.load_state_dict(state_dict)
    model.eval()
    for j, (test_X, test_Y) in enumerate(all_loader):
        test_X = test_X
        test_Y = test_Y
        LSTM_pre_all = model(test_X)  # [32,1,1]
        LSTM_loss = criterion(LSTM_pre_all, test_Y)  # [1,]
        print("全部数据的LSTM预测loss=", LSTM_loss.item())  # 保存loss

    # 绘制曲线
    fig_name = "LSTMNet_y"
    fontsize = 15
    x_range = dataset[0:2800, 0]
    true_y = dataset[0:2800, -1]
    pre_y = LSTM_pre_all.squeeze(2).detach().numpy()  # [2797,4]
    pre_y = LSTM_mean(pre_y)
    plt.xlabel("t/s")  # 给x轴起名字
    plt.ylabel("y")  # 给y轴起名字
    plt.plot(x_range, true_y, label='true_y(t)')
    plt.plot(x_range, pre_y, label='LSTMnet_y(t)')
    plt.legend()
    plt.tight_layout()
    plt.savefig('./results/' + fig_name + '.png')
    plt.show()

    print("finished")




