#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2022/7/1 13:37
# @Author  : lanlin


import torch
import torch.nn.functional as Fun
import numpy as np
import matplotlib.pyplot as plt
from dataset import dataset, data_train, data_test


lr = 0.01  # 学习率
epochs = 500
n_feature = 3  # 特征维度为3的话，就是包含时间位置信息
hiddens = [200, 1000, 200]  # 隐藏层的层数及神经元数量
n_output = 1


# 定义BP神经网络
class BPNetModel(torch.nn.Module):
    def __init__(self, n_feature, hiddens, n_output):
        super(BPNetModel, self).__init__()
        self.hiddden1 = torch.nn.Linear(n_feature, hiddens[0])  # 定义第1隐层网络
        self.hiddden2 = torch.nn.Linear(hiddens[0], hiddens[1])  # 定义第2隐层网络
        self.hiddden3 = torch.nn.Linear(hiddens[1], hiddens[2])  # 定义第3隐层网络
        self.dropout = torch.nn.Dropout(p=0.1)  # p=0.1表示神经元有p = 0.1的概率不被激活
        self.BN1 = torch.nn.BatchNorm1d(num_features=hiddens[0])
        self.BN2 = torch.nn.BatchNorm1d(num_features=hiddens[1])
        self.out = torch.nn.Linear(hiddens[2], n_output)  # 定义输出层网络

    def forward(self, x):

        x = Fun.relu(self.hiddden1(x))  # 隐层激活函数采用relu()函数
        x = self.BN1(x)
        x = self.dropout(x)
        x = Fun.relu(self.hiddden2(x))
        x = self.BN2(x)
        x = self.dropout(x)
        x = Fun.relu(self.hiddden3(x))
        out = self.out(x)

        return out


if __name__ == "__main__":

    # 定义优化器和损失函数
    net = BPNetModel(n_feature=n_feature, hiddens=hiddens, n_output=n_output)  # 初始化网络
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)  # 使用Adam优化器，并设置学习率
    loss_fun = torch.nn.MSELoss()

    # 将数据类型转换为tensor
    x_train = torch.tensor(data_train[:,0:3])
    y_train = torch.tensor(data_train[:,-1]).unsqueeze(1)
    x_test = torch.tensor(data_test[:,0:3])
    y_test = torch.tensor(data_test[:,-1]).unsqueeze(1)

    # 训练数据
    train_loss_array = np.zeros(epochs)  # 构造一个array([ 0., 0., 0., 0., 0.])里面有epochs个0
    test_loss_array = np.zeros(epochs)

    for epoch in range(epochs):
        net.train()
        train_pred = net(x_train)  # 前向传播
        train_loss = loss_fun(train_pred, y_train)  # 预测值和真实值对比
        optimizer.zero_grad()  # 梯度清零
        train_loss.backward()  # 反向传播
        optimizer.step()  # 更新梯度
        train_loss_array[epoch] = train_loss.item()  # 保存loss
        running_loss = train_loss.item()

        net.eval()  # 下面是没有梯度的计算,主要是测试集使用，不需要再计算梯度了
        test_pred = net(x_test)
        test_loss = loss_fun(test_pred, y_test)
        test_loss_array[epoch] = test_loss

        # 打印训练参数
        if epoch % 100 == 0:
            print('Epoch: {:4}, train Loss: {:.5f}, Test Loss: {:.5f}'
                  .format(epoch, train_loss.item(), test_loss.item()))

    torch.save(net.state_dict(), './weights/BP_params.pth')

    # 绘制损失函数和精度
    fig_name = "BPNet"
    fontsize = 15
    fig, (ax1, ax2) = plt.subplots(2, figsize=(15, 12), sharex=True)
    ax1.plot(train_loss_array)
    ax1.set_ylabel("train loss", fontsize=fontsize)
    ax1.set_title(fig_name, fontsize="xx-large")
    ax2.plot(test_loss_array)
    ax2.set_ylabel("test loss", fontsize=fontsize)
    ax2.set_xlabel("epochs", fontsize=fontsize)
    plt.tight_layout()
    plt.savefig('./results/' + fig_name + '.png')
    plt.show()

    # 与原y(t)图像对比
    model = BPNetModel(n_feature=n_feature, hiddens=hiddens, n_output=n_output)
    model.eval()
    model_paras_path = './weights/BP_params.pth'
    state_dict = torch.load(model_paras_path)
    model.load_state_dict(state_dict)
    x_all = torch.tensor(dataset[:, 0:3])
    y_all = torch.tensor(dataset[:, -1]).unsqueeze(1)
    BP_pre_all = model(x_all)
    BP_loss = loss_fun(BP_pre_all, y_all)  # 预测值和真实值对比
    print("全部数据的BP预测loss=", BP_loss.item())

    fig_name = "BPNet_y"
    fontsize = 15
    x_range = dataset[:, 0]
    true_y = dataset[:, -1]
    pre_y = BP_pre_all.detach().numpy()[:, 0]
    plt.xlabel("t/s")  # 给x轴起名字
    plt.ylabel("y")  # 给y轴起名字
    plt.plot(x_range, true_y, label='true_y(t)')
    plt.plot(x_range, pre_y, label='BPnet_y(t)')
    plt.legend()
    plt.tight_layout()
    plt.savefig('./results/' + fig_name + '.png')
    plt.show()
