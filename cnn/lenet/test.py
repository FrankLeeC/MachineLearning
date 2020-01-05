# -*-coding:utf-8-*-

# 测试文件

import numpy as np
import model

def test_conv():
    a = np.array([[1, 1, 1, 1, 1, 1], [2, 2, 2, 2, 2, 2], [3, 3, 3, 3, 3, 3], [4, 4, 4, 4, 4, 4], [5, 5, 5, 5, 5, 5], [6, 6, 6, 6, 6, 6]])
    f = np.array([[[1, 1, 1], [1, 1, 1], [1, 1, 1]], [[2, 2, 2], [2, 2, 2], [2, 2, 2]], [[3, 3, 3], [3, 3, 3], [3, 3, 3]]])
    print(a)
    print('---------')
    print(f)
    print('----------------')
    c = model.convolute2d(a, f)
    print(c.shape)  # (3, 4, 4)
    print(c[0])

def test_pooling():
    a = np.array([[1, 1, 1, 1, 1, 1], [2, 2, 2, 2, 2, 2], [3, 3, 3, 3, 3, 3], [4, 4, 4, 4, 4, 4], [5, 5, 5, 5, 5, 5], [6, 6, 6, 6, 6, 6]])
    b = model.pooling(a, 3, 2, padding=model.PADDING_SAME, mode=model.POOLING_MAX)
    print(b)


if __name__ == "__main__":
    # convolute()
    # image()
    # list_join()
    # test_conv()
    test_pooling()