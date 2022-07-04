#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2022/6/28 11:10
# @Author  : lanlin


import numpy as np
import matplotlib.pyplot as plt



def input_I(t):
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
    return I


def input_u(t):
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

    return u


def output_y(t, y1, y2):
    """
    :param t: 时间t，从0开始
    :param y1: 输出y在t-1时刻的值
    :param y2: 输出y在t-2时刻的值
    :param u1: 输入u在t-1时刻的值
    :param u2: 输入u在t-2时刻的值
    :param I2: 输入I在t-2时刻的值
    :return: 返回t时刻的系统响应y，0时刻之前响应为0
    """
    I2 = input_I(t-2)
    u1 = input_u(t-1)
    u2 = input_u(t-2)
    y = 0.4*y1 - 0.3*y1*u1/u2 + 0.3*1.6*y2*u1/u2 + \
        0.3*2.0*u1*I2 - 0.3*u1*y1 + 0.3*1.6*u1*y2

    return y


data_array = []
y_before = [0.0, 0.0, 0.0, 0.0]  # 时间往左向后推进，每0.5秒一个值
t = 0.0  # 共2801个数据对
while t < 1400.5:
    y_t = output_y(t,y_before[2],y_before[0])  # y_befor[0]代表t-2
    data_row = [t, input_I(t), input_u(t), y_t]
    data_array.append(data_row)
    y_before.pop(0)
    y_before.append(y_t)
    t += 0.5

dataset = np.array(data_array)

# 随机生成2000训练集和400测试集
data_index = [i for i in range(0,2801)]
used_data_index = np.random.choice(data_index, 2400, replace=False)
used_test_index = np.random.choice(used_data_index, 400, replace=False)
data_train = []
data_test = []
for i in range(0, 2801):
    data_temp = data_array[i]
    if data_temp==[]:
        print("数组为空")
    if i in used_test_index:
        data_test.append(data_temp)
    elif i in used_data_index:
        data_train.append(data_temp)
    else:
        continue

data_train = np.array(data_train).astype(np.float32)
data_test = np.array(data_test).astype(np.float32)


if __name__=="__main__":

    fontsize = 15
    fig, (ax1, ax2, ax3) = plt.subplots(3, figsize=(15, 12), sharex=True)
    ax1.plot(dataset[:,0], dataset[:,1])
    ax1.set_ylabel("input_I(t)", fontsize=fontsize)
    ax1.set_title("function_figure", fontsize="xx-large")
    ax2.plot(dataset[:,0], dataset[:,2])
    ax2.set_ylabel("input_u(t)", fontsize=fontsize)
    ax3.plot(dataset[:,0], dataset[:,3])
    ax3.set_ylabel("output_y(t)", fontsize=fontsize)
    ax3.set_xlabel("t/s", fontsize=fontsize)
    plt.tight_layout()
    plt.savefig('function_figure.png')
    plt.show()

    print("dataset finished")