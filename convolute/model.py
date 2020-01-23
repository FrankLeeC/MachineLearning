# -*-coding:utf-8-*-

import numpy as np


# PADDING MODE
PADDING_SAME = 0  # 与输出一致
PADDING_VALID = 1  # 不做对齐补充  会有损失

# POOLING MODE
POOLING_MAX = 0  # 最大
POOLING_AVG = 1  # 平均
POOLING_ECHO = 2  # 原样返回

# 2维卷积
# matrix的层数与filters层数一致
# matrix  单个正方形图层通道
# filters 过滤器集合  可以多个过滤器
# bias 偏移量 集合
# stride filter 移动步长
# padding 对齐方式
# 返回与过滤器个数相同的卷积层
def convolute2d(matrix, filters, bias, stride, padding=PADDING_SAME):
    fsize, _ = filters[0].shape
    matrix, rounds = __padding(matrix, fsize, stride, padding)
    result = []
    for i, f in enumerate(filters):
        f_width, _ = f.shape
        matrix.shape
        d = [[__conv2d(matrix[r*stride:r*stride+f_width,c*stride:c*stride+f_width], f, bias[i]) for c in range(rounds)] for r in range(rounds)]
        result.append(d)
    return np.array(result)

def __conv2d(m, f, b):
    return np.sum(m*f) + b

# matrix 正方形图层通道 集合
# fsize 窗口大小
# bias 偏移量
# round = [(n+2*p-f)//s]+1  round == out_width
# PADDING_SAME    round = out_width = n  p = [s*(n-1)+f-n]/2
# PADDING_VALID       p = 0   round == out_width = [(n-fsize)//stride]+1
def pooling(matrix, fsize, bias, stride, padding=PADDING_SAME, mode=POOLING_MAX):
    result = []
    for _, data in enumerate(matrix):
        data, rounds = __padding(data, fsize, stride, padding)
        f = __max
        if mode == POOLING_AVG:
            f = __avg
        if mode == POOLING_ECHO:
            f = __echo
        result.append([[f(data[r*stride:r*stride+fsize,c*stride:c*stride+fsize])+bias for c in range(rounds)] for r in range(rounds)])
    return np.array(result)

def __max(matrix):
    return np.max(matrix)

def __avg(matrix):
    return np.average(matrix)

def __echo(matrix):
    return matrix

def __padding(data, fsize, stride, padding=PADDING_SAME):
    n, _ = data.shape
    if padding == PADDING_VALID:
        return data, ((n-fsize)//stride)+1
    a = stride*(n-1)+fsize-n
    pad_left, pad_right = 0, 0
    if a % 2 == 0:
        pad_left = pad_right = a / 2
    else:
        pad_left = a // 2
        pad_right = a - pad_left
    result = np.zeros([pad_left + n + pad_right, pad_left + n + pad_right])
    result[pad_left:pad_left+n,pad_left:pad_left+n] = data
    return result, n
