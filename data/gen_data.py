# -*-encoding=utf-8-*-

import random

import numpy as np


def quadrant(count):
    data = []
    label = []
    for i in range(count):
        x1 = random.uniform(-1, 1)
        x2 = random.uniform(-1, 1)
        data.append([x1, x2])
        if (x1 >= 0 and x2 >= 0) or (x1 < 0 and x2 < 0):
            label.append([1.0])
        else:
            label.append([0.0])
    return np.array(data), np.array(label)
