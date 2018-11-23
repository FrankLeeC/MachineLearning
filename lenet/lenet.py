# -*- coding: utf-8 -*-

import numpy as np

def convolute(input, filter):
    w, h = np.shape(input)
    w1, h1 = np.shape(filter)
    w2 = w - w1 + 1
    h2 = h - w1 + 1
    output = np.zeros([w2, h2])
    for i in range(w2):
        for j in range(h2):
            tmp_input = input[i:w1+i, j:h1+j]
            output[i][j] = np.sum(tmp_input * filter)
    return output


def max_pooling(input, size):
    w, h = np.shape(input)
    w1, h1 = size
    w2 = w - w1 + 1
    h2 = h - w1 + 1
    output = np.zeros([w2, h2])
    for i in range(w2):
        for j in range(h2):
            tmp_input = input[i:w1+i, j:h1+j]
            output[i][j] = np.max(tmp_input)
    return output

a = np.array([
    [1, 2, 3, 4, 5, 3],
    [1, 0, 2, 3, 2, 1],
    [2, 0, 1, 0, 3, 3],
    [4, 1, 0, 2, 1, 7],
    [3, 1, 3, 2, 0, 2]
])

b = np.array([
    [1, 0, 1],
    [1, 0, 1],
    [1, 0, 1]
])

c = convolute(a, b)
print(c)
c = max_pooling(a, [3, 3])
print(c)
