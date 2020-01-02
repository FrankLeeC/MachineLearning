# -*-coding:utf-8-*-

import numpy as np

# 卷积
# matrix的层数与filters层数一致
# matrix  正方形图层
# filters 过滤器集合  可以多个过滤器
# padding 对齐
def convolute(matrix, filters):
    result = []
    for _, f in enumerate(filters):
        f_width, _ = f.shape
        m_width, _, _ = matrix.shape
        rounds = m_width - f_width + 1
        d = [[conv2d(matrix[c:c+f_width,r:r+f_width,:], f) for c in range(rounds)] for r in range(rounds)]
        result.append(d)
    return np.array(result)

def conv2d(m, f):
    return np.sum(m*f)