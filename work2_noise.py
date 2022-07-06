#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2022/7/1 13:38
# @Author  : lanlin


import torch
import numpy as np
import matplotlib.pyplot as plt
from work1_BP import BPNetModel,n_feature,hiddens,n_output


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


if __name__=="__main__":

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
    dataset_noised = np.array(data_array).astype(np.float32)
    dataset_noised = torch.tensor(dataset_noised)

    # read model
    loss_fun = torch.nn.MSELoss()
    model = BPNetModel(n_feature=n_feature, hiddens=hiddens, n_output=n_output)
    model.eval()
    model_paras_path = './weights/BP_params.pth'
    state_dict = torch.load(model_paras_path)
    model.load_state_dict(state_dict)
    pre_noised = model(dataset_noised[:,0:3])
    loss_noised = loss_fun(pre_noised, dataset_noised[:,-1])  # 预测值和真实值对比
    print("添加噪声后的的预测loss", loss_noised.item())

    # 绘制曲线
    fontsize = 15
    fig, (ax1, ax2, ax3) = plt.subplots(3, figsize=(15, 12), sharex=True)
    ax1.plot(dataset_noised[:, 0], dataset_noised[:, 1])
    ax1.set_ylabel("noised_I(t)", fontsize=fontsize)
    ax1.set_title("noised_function_figure", fontsize="xx-large")
    ax2.plot(dataset_noised[:, 0], dataset_noised[:, 2])
    ax2.set_ylabel("noised_u(t)", fontsize=fontsize)
    ax3.set_ylabel("y(t)", fontsize=fontsize)
    ax3.set_xlabel("t/s", fontsize=fontsize)
    ax3.plot(dataset_noised[:, 0].detach().numpy(), dataset_noised[:, 3].detach().numpy(), label='expected_y(t)')
    ax3.plot(dataset_noised[:, 0].detach().numpy(), pre_noised.detach().numpy(), label='BPnet_y(t)')
    plt.legend()
    plt.tight_layout()
    plt.savefig('./results/noised_function_figure.png')
    plt.show()