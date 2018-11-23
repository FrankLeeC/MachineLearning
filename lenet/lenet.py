# -*- coding: utf-8 -*-

import numpy as np

def convolute(input, filter):
    '''
    width == height
    '''
    w, _ = np.shape(input)
    w1, _ = np.shape(filter)
    w2 = w - w1 + 1
    output = np.zeros([w2, w2])
    for i in range(w2):
        for j in range(w2):
            tmp_input = input[i:w1+i, j:w1+j]
            output[i][j] = np.sum(tmp_input * filter)
    return output


def max_sampling(input, size):
    '''
    width == height
    size < width
    '''
    w, _ = np.shape(input)
    w2 = w - size + 1
    output = np.zeros([w2, w2])
    for i in range(w2):
        for j in range(w2):
            tmp_input = input[i:size+i, j:size+j]
            output[i][j] = np.max(tmp_input)
    return output

a = np.array([
    [1, 2, 3, 4, 5],
    [1, 0, 2, 3, 2],
    [2, 0, 1, 0, 3],
    [4, 1, 0, 2, 1],
    [3, 1, 3, 2, 0]
])

b = np.array([
    [1, 0, 1],
    [1, 0, 1],
    [1, 0, 1]
])

c = convolute(a, b)
print(c)
c = max_sampling(a, 3)
print(c)
