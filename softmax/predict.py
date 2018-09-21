# -*-encode=utf-8-*-

import numpy as np
from scipy.misc import imread


theta = np.load('./theta68.txt')
image = imread('./nine.png')
image = image.reshape(image.shape[0] * image.shape[1], 1)
a = np.exp(theta * image)
b = np.sum(np.exp(theta * image), axis=0)
r = a / b
max = r[0]
index = 0
sum = 0
for i, e in enumerate(r):
    sum += e
    if e > max:
        max = e
        index = i
print(sum)
print('---')
print(r)
print('digit is:%d' % index)