# -*- coding:utf-8 -*-

from PIL import Image
import numpy as np
import os

'''
0 black
255 white
'''

# shape [1, x*y]
def __convert(fname):
    m = np.where(np.array(Image.open(fname))>=50, 255, 0)
    x, y = m.shape
    m = np.reshape(m, (1, x * y))
    return m

def load(path):
    labels = os.listdir(path)
    x, y = [], []
    for _, l in enumerate(labels):
        files = os.listdir(path+l+"/")
        for _, f in enumerate(files):
            x.append(__convert(path+l+"/"+f))
            y.append(int(l))
    return x, y


def load_training_set():
    return load('/Users/frank/Pictures/mnist/train/data_gray/')
    
def load_test_set():
    return load('/Users/frank/Pictures/mnist/test/data_gray/')
