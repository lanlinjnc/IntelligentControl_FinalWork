#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2022/7/1 13:38
# @Author  : lanlin


import torch
import numpy as np
import matplotlib.pyplot as plt
from work1_BP import BPNetModel,n_feature,hiddens,n_output
from work3_LSTM import MyLSTM, Mydataset, input_size, output_size, mid_dim, mid_layers
from work3_LSTM import data_index as LSTM_data_index


normal_mu = 0.0
normal_sigma = 0.01
noise_num = int(2801*0.15)
data_index = [i for i in range(0, 2801)]
np.random.seed(20220705)
used_data_index = np.random.choice(data_index, noise_num, replace=False)  # 抽取坐标


def input_noised_I(t):
    """
    :param t: 时间t，[-2,1400]
    :return: 系统输入I，0时刻之前为0
    """
    I = 0.0
    if -2 <= t < 0:
        I = 0.0
    elif 0 <= t <= 1400:
        I = np.sin(np.pi * t / 1400)
    else:
        print("I(t)时间范围错误")
    if int(t*2) in used_data_index:
        noise = np.random.normal(loc=normal_mu, scale=normal_sigma, size=1)
        if noise > 0.1:  # 防止噪声太大影响函数图像展示
            noise = 0.1
        return I + noise
    else:
        return I


def input_noised_u(t):
    """
    :param t: 时间t，[-2,1400]
    :return: 系统输入u，0时刻之前为0
    """
    u = 0
    if -2 <= t <= 350:
        u = 0.175
    elif 350 < t <= 525:
        u = 0.25
    elif 525 < t <= 877:
        u = 0.5
    elif 877 < t <= 1052:
        u = 0.25
    elif 1052 < t <= 1400:
        u = 0.175
    else:
        print("u(t)时间范围错误")

    if int(t*2) in used_data_index:
        noise = np.random.normal(loc=normal_mu, scale=normal_sigma, size=1)
        if noise > 0.1:  # 防止噪声太大影响函数图像展示
            noise = 0.1
        return u + noise
    else:
        return u


def output_noised_y(t, y1, y2):
    """
    :param t: 时间t，从0开始
    :param y1: 输出y在t-1时刻的值
    :param y2: 输出y在t-2时刻的值
    :param u1: 输入u在t-1时刻的值
    :param u2: 输入u在t-2时刻的值
    :param I2: 输入I在t-2时刻的值
    :return: 返回t时刻的系统响应y，0时刻之前响应为0
    """
    I2 = input_noised_I(t-2)
    u1 = input_noised_u(t-1)
    u2 = input_noised_u(t-2)
    y = 0.4*y1 - 0.3*y1*u1/u2 + 0.3*1.6*y2*u1/u2 + \
        0.3*2.0*u1*I2 - 0.3*u1*y1 + 0.3*1.6*u1*y2

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

    # 生成噪声数据
    data_array = []
    y_before = [0.0, 0.0, 0.0, 0.0]  # 时间往左向后推进，每0.5秒一个值
    t = 0.0  # 共2801个数据对
    while t < 1400.5:
        y_t = output_noised_y(t, y_before[2], y_before[0])  # y_befor[0]代表t-2
        data_row = [t, input_noised_I(t), input_noised_u(t), y_t]
        data_array.append(data_row)
        y_before.pop(0)
        y_before.append(y_t)
        t += 0.5
    dataset_numpy = np.array(data_array, dtype=np.float32)
    dataset_noised = torch.tensor(dataset_numpy)

    # read BP model
    loss_fun = torch.nn.MSELoss()
    model = BPNetModel(n_feature=n_feature, hiddens=hiddens, n_output=n_output)
    model.eval()
    model_paras_path = './weights/BP_params.pth'
    state_dict = torch.load(model_paras_path)
    model.load_state_dict(state_dict)
    BP_pre_noised = model(dataset_noised[:,0:3])
    BP_loss = loss_fun(BP_pre_noised, dataset_noised[:,-1])  # 预测值和真实值对比
    print("添加噪声后的的BP预测loss=", BP_loss.item())

    # read LSTM model
    data_test = Mydataset(dataset_numpy, LSTM_data_index)
    noise_loader = torch.utils.data.DataLoader(data_test, batch_size=3000)

    model = MyLSTM(input_size, output_size, mid_dim, mid_layers)
    model_paras_path = './weights/LSTM_params.pth'
    state_dict = torch.load(model_paras_path)
    model.load_state_dict(state_dict)
    model.eval()
    for j, (test_X, test_Y) in enumerate(noise_loader):
        test_X = test_X
        test_Y = test_Y
        LSTM_pre_noised = model(test_X)  # [32,1,1]
        LSTM_loss = loss_fun(LSTM_pre_noised, test_Y)  # [1,]
        print("添加噪声后的的LSTM预测loss=", LSTM_loss.item())  # 保存loss

    # 绘制曲线
    x_range = dataset_noised[0:2800, 0].detach().numpy()
    BP_y = BP_pre_noised[0:2800, 0].detach().numpy()  # 用于观察数据
    LSTM_y = LSTM_pre_noised.squeeze(2).detach().numpy()  # [2797,4]
    LSTM_y = LSTM_mean(LSTM_y)  # (2800,) 将LSTM_y的二维数据根据LSTM特性求均值变为一维数据
    fontsize = 15
    fig, (ax1, ax2, ax3) = plt.subplots(3, figsize=(15, 12), sharex=True)
    ax1.plot(x_range, dataset_noised[0:2800, 1])
    ax1.set_ylabel("noised_I(t)", fontsize=fontsize)
    ax1.set_title("noised_function_figure", fontsize="xx-large")
    ax2.plot(x_range, dataset_noised[0:2800, 2])
    ax2.set_ylabel("noised_u(t)", fontsize=fontsize)
    ax3.set_ylabel("y(t)", fontsize=fontsize)
    ax3.set_xlabel("t/s", fontsize=fontsize)
    ax3.plot(x_range, dataset_noised[0:2800,3].detach().numpy(), label='expected_y(t)')
    ax3.plot(x_range, BP_y, label='BPnet_y(t)')
    ax3.plot(x_range, LSTM_y, label='LSTMnet_y(t)')
    plt.legend()
    plt.tight_layout()
    plt.savefig('./results/noised_function_figure.png')
    plt.show()