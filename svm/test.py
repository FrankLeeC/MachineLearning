import svm
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse, Circle
import random
import math

# rbf
def get_rbf_training_examples():
    X1, X2 = [], []
    y1, y2 = [], []
    for i in range(110):
        x = -5.5 + 0.1 * i
        if x < -5 or x > 5:
            X1.append([x, random.random() * 6 - 3])
            y1.append(1)
        else:
            f = math.sqrt(25-x**2)
            fm = -f
            if i % 2 == 0:
                y = f + random.random() *  6 - 3
            else:
                y = fm + random.random() *  6 - 3
            if (y > f or y < fm):
                X1.append([x, y])
                y1.append(-1)
            elif (y < f and y > fm):
                X2.append([x, y])
                y2.append(-1)
    return X1, y1, X2, y2

# rbf
def get_rbf_test_examples():
    X1, X2 = [], []
    y1, y2 = [], []
    for i in range(110):
        x = -5.5 + 0.1 * i
        if x < -5 or x > 5:
            X1.append([x, random.random() * 6 - 3])
            y1.append(1)
        else:
            f = math.sqrt(25-x**2)
            fm = -f
            if i % 2 == 0:
                y = f + random.random() * 6 - 3
            else:
                y = fm + random.random() * 6 - 3
            if (y > f or y < fm):
                X1.append([x, y])
                y1.append(-1)
            elif (y < f and y > fm):
                X2.append([x, y])
                y2.append(-1)
    return X1, y1, X2, y2

# polynomial
def get_polynomial_training_examples():
    X1, X2 = [], []
    y1, y2 = [], []
    for i in range(50):
        x = 0.1 + 0.1 * i
        f = 1.0 / x
        y = f + random.random() * 8 - 4
        if y < f:
            X1.append([x, y])
            y1.append(1)
        elif y > f:
            X2.append([x, y])
            y2.append(-1)
    return X1, y1, X2, y2

# polynomia
def get_polynomial_test_examples():
    X1, X2 = [], []
    y1, y2 = [], []
    for i in range(20):
        x = 0.1 + 0.1 * i
        f = 1.0 / x
        y = f + random.random() * 8 - 4
        if y < f:
            X1.append([x, y])
            y1.append(1)
        elif y > f:
            X2.append([x, y])
            y2.append(-1)
    return X1, y1, X2, y2

# outlier
def get_linear_outlier_training_examples():
    X1 = np.array([[8, 7], [4, 10], [9, 7], [7, 10],
                   [9, 6], [4, 8], [10, 10]])
    y1 = np.ones(len(X1))
    X2 = np.array([[2, 7], [8, 3], [7, 5], [4, 4],
                   [7, 8],  # the outlier
                   [4, 6], [1, 3], [2, 5]])
    y2 = np.ones(len(X2)) * -1
    return X1, y1, X2, y2

# outlier
def get_linear_outlier_test_examples():
    X1 = np.array([[2, 9], [1, 10], [1, 11], [3, 9], [11, 5],
                   [10, 6], [10, 11], [7, 8], [8, 8], [4, 11],
                   [9, 9], [7, 7], [11, 7], [5, 8], [6, 10]])
    X2 = np.array([[11, 2], [11, 3], [1, 7], [5, 5], [6, 4],
                   [9, 4], [2, 6], [9, 3], [7, 4], [7, 2], [4, 5],
                   [3, 6], [1, 6], [2, 3], [1, 1], [4, 2], [4, 3]])
    y1 = np.ones(len(X1))
    y2 = np.ones(len(X2)) * -1
    return X1, y1, X2, y2

# linear
def get_linear_training_examples():
    X1 = np.array([[8, 7], [4, 10], [9, 7], [7, 10],
                   [9, 6], [4, 8], [10, 10]])
    y1 = np.ones(len(X1))
    X2 = np.array([[2, 7], [8, 3], [7, 5], [4, 4],
                   [4, 6], [1, 3], [2, 5]])
    y2 = np.ones(len(X2)) * -1
    return X1, y1, X2, y2

# linear
def get_linear_test_examples():
    X1 = np.array([[2, 9], [1, 10], [1, 11], [3, 9], [11, 5],
                   [10, 6], [10, 11], [7, 8], [8, 8], [4, 11],
                   [9, 9], [7, 7], [11, 7], [5, 8], [6, 10]])
    X2 = np.array([[11, 2], [11, 3], [1, 7], [5, 5], [6, 4],
                   [9, 4], [2, 6], [9, 3], [7, 4], [7, 2], [4, 5],
                   [3, 6], [1, 6], [2, 3], [1, 1], [4, 2], [4, 3]])
    y1 = np.ones(len(X1))
    y2 = np.ones(len(X2)) * -1
    return X1, y1, X2, y2

def linear():
    x1, y1, x2, y2 = get_linear_outlier_training_examples()
    x = np.vstack((x1, x2))
    y = np.hstack((y1, y2))
    model = svm.SVM(kernel='linear', C=0.0)
    model.fit(x, y)
    for p in x1:
        plt.scatter(p[0], p[1], c='blue', marker='o', alpha=1, edgecolors='none')
    for p in x2:
        plt.scatter(p[0], p[1], c='red', marker='o', alpha=1, edgecolors='none')    
    w, b = model.get_model()
    print(w)
    print(b)

    k = float(-w[0:1,0:1]/w[0:1,1:2])
    intercept = float(b/w[0:1,1:2])
    print(k, intercept)
    p1 = [0, 10]
    p2 = [float(b), k*10+intercept]
    plt.plot(p1, p2, c='black')
  
    x1, y1, x2, y2 = get_linear_outlier_test_examples()
    x = np.vstack((x1, x2))
    y = np.hstack((y1, y2))
    succ = 0
    total = 0
    s_set = set()
    for i in range(x.shape[0]):
        total += 1
        pred = model.predict(x[i])
        if pred == y[i]:
            s_set.add(i)
            succ += 1
    print('accuracy:', succ / total)

    c = 0
    for p in x1:
        if c in s_set:
            plt.scatter(p[0], p[1], c='blue', marker='^', alpha=1, edgecolors='none')
        else:
            plt.scatter(p[0], p[1], c='black', marker='^', alpha=1, edgecolors='none')
        c += 1
    for p in x2:
        if c in s_set:
            plt.scatter(p[0], p[1], c='red', marker='^', alpha=1, edgecolors='none')
        else:
            plt.scatter(p[0], p[1], c='black', marker='^', alpha=1, edgecolors='none')
        c += 1

    plt.grid(True)
    plt.show(block=True)

def polynomial():
    x1, y1, x2, y2 = get_polynomial_training_examples()
    x = np.vstack((x1, x2))
    y = np.hstack((y1, y2))
    model = svm.SVM(kernel='polynomial', C=0.0)
    model.fit(x, y)
    for p in x1:
        plt.scatter(p[0], p[1], c='blue', marker='o', alpha=1, edgecolors='none')
    for p in x2:
        plt.scatter(p[0], p[1], c='red', marker='o', alpha=1, edgecolors='none')    
    w, b = model.get_model()
    print(w)
    print(b)
  
    x1, y1, x2, y2 = get_polynomial_test_examples()
    x = np.vstack((x1, x2))
    y = np.hstack((y1, y2))
    succ = 0
    total = 0
    s_set = set()
    for i in range(x.shape[0]):
        total += 1
        pred = model.predict(x[i])
        if pred == y[i]:
            s_set.add(i)
            succ += 1
    print('accuracy:', succ / total)

    c = 0
    for p in x1:
        if c in s_set:
            plt.scatter(p[0], p[1], c='blue', marker='^', alpha=1, edgecolors='none')
        else:
            plt.scatter(p[0], p[1], c='black', marker='^', alpha=1, edgecolors='none')
        c += 1
    for p in x2:
        if c in s_set:
            plt.scatter(p[0], p[1], c='red', marker='^', alpha=1, edgecolors='none')
        else:
            plt.scatter(p[0], p[1], c='black', marker='^', alpha=1, edgecolors='none')
        c += 1

    plt.grid(True)
    plt.show(block=True)

def rbf():
    x1, y1, x2, y2 = get_rbf_training_examples()
    x = np.vstack((x1, x2))
    y = np.hstack((y1, y2))
    model = svm.SVM(kernel='rbf', C=0.0)
    model.fit(x, y)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    circle = Circle(xy=(0.0, 0.0), radius=5, alpha=0.3)
    ax.add_patch(circle)
    for p in x1:
        plt.scatter(p[0], p[1], c='blue', marker='o', alpha=1, edgecolors='none')
    for p in x2:
        plt.scatter(p[0], p[1], c='red', marker='o', alpha=1, edgecolors='none')    
    w, b = model.get_model()
    print(w)
    print(b)

    x1, y1, x2, y2 = get_rbf_test_examples()
    x = np.vstack((x1, x2))
    y = np.hstack((y1, y2))
    succ = 0
    total = 0
    s_set = set()
    for i in range(x.shape[0]):
        total += 1
        pred = model.predict(x[i])
        if pred == y[i]:
            s_set.add(i)
            succ += 1
    print('accuracy:', succ / total)

    c = 0
    for p in x1:
        if c in s_set:
            plt.scatter(p[0], p[1], c='blue', marker='^', alpha=1, edgecolors='none')
        else:
            plt.scatter(p[0], p[1], c='black', marker='^', alpha=1, edgecolors='none')
        c += 1
    for p in x2:
        if c in s_set:
            plt.scatter(p[0], p[1], c='red', marker='^', alpha=1, edgecolors='none')
        else:
            plt.scatter(p[0], p[1], c='black', marker='^', alpha=1, edgecolors='none')
        c += 1

    plt.grid(True)
    plt.show(block=True)

def main():
    # linear()
    polynomial()
    # rbf()


if __name__ == '__main__':
    main()