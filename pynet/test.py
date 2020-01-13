# -*- coding:utf-8 -*-

from PIL import Image
import numpy as np
import os
import loss
import network
import neural
import optimizer

def __convert(fname):
    m = np.where(np.array(Image.open(fname))>=50, 255, 0)/2000
    x, y = m.shape
    m = np.reshape(m, (1, x * y))
    return m

def load(path):
    labels = ['0', '1']
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

def get_network():
    n = network.NetWork('network')
    n.add(neural.Dense(1024, input_dim=784))
    n.add(neural.Relu(0.0))
    n.add(neural.Dense(512))
    n.add(neural.Relu(0.0))
    n.add(neural.Dense(1))
    n.add(neural.Sigmoid())
    n.loss(loss.BinaryCrossEntropy())
    n.optimizer(optimizer.Adagrad())
    return n

def run():
    x, y = load_training_set()
    x, y = x[0:1], y[0:1]
    n = get_network()
    n.train(x, y, 1, iteration=5000)
    x, y = load_test_set()
    x, y = x[0:1], y[0:1]
    y1 = n.predict(x)
    print(y*np.log(y1))
    
if __name__ == "__main__":
    run()