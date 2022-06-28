#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2022/6/28 11:10
# @Author  : lanlin


import numpy as np
import torch



def input_I(t):
    """
    :param t: 时间t，[-2,1400]
    :return: 系统输入I，0时刻之前为0
    """
    I = 0.0
    if -2 <= t <= 1400:
        I = np.sin(np.pi * t / 1400)
    else:
        print("时间范围错误")
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
        u = 0.25
    else:
        print("时间范围错误")

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



if __name__=="__main__":
    pass
