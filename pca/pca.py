import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.datasets.samples_generator import make_blobs

def random2():
    x, y = make_blobs(n_samples=160, n_features=3, centers=[[3, 3, 3], [0, 0, 0], [1, 1, 1], [2, 2, 2]], cluster_std=[0.2, 0.1, 0.2, 0.2])
    return x

def random():
    a = [[1, 1, 1], 
        [1, -1, 1], 
        [-1, -1, 1], 
        [-1, 1, 1],
        [1, 1, -1],
        [1, -1, -1],
        [-1, -1, -1],
        [-1, 1, -1]]
    r = None
    for i in range(20):
        if r is None:
            r = a*np.random.randint(0, 10, (1, 3))
        else:
            r = np.concatenate((r,a*np.random.randint(0, 10, (1, 3))), axis=0)
    return r

def main():
    x = random2()
    plot(x)
    mean = np.sum(x, axis=0)/np.shape(x)[0]
    x2 = x - mean
    xcov = np.cov(x, rowvar=False)
    # print(y)
    a, b = np.linalg.eig(xcov)
    m = np.argmin(a)
    d = []
    for i, v in enumerate(a):
        if i != m:
            d.append(b[i])
    print(a, '----')
    print(b, '----')
    print(d, '----')
    new = np.dot(x2, np.transpose(d))
    # print(new)
    plot2d(new, 'blue', False)

    new2 = sklearn(x)
    plot2d(new2, 'red', True)

def sklearn(data):
    pca = PCA(n_components=2)
    pca.fit(data)
    print(pca.explained_variance_, '=====')
    print(pca.explained_variance_ratio_, '====')
    return pca.transform(data)

def plot2d(data, color, b):
    fig, ax = plt.subplots()
    for d in data:
        ax.scatter(d[0], d[1], c=color, alpha=1, edgecolors='none')
    plt.show(block=b)

def plot(data):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    c = 'blue'
    for d in data:
        # if d[0] >= 0 and d[1] >= 0:
        #     c = 'green'
        # elif d[0] >= 0 and d[1] < 0:
        #     c = 'black'
        # elif d[0] < 0 and d[1] < 0:
        #     c = 'orange'
        # elif d[0] < 0 and d[1] >= 0:
        #     c = 'yellow'
        ax.scatter(d[0], d[1], d[2], c=c)
    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')
    plt.show(block=False)

if __name__ == '__main__':
    main()