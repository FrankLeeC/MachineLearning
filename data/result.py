# -*-encoding=utf-8-*-
import matplotlib.pyplot as plt


class Result:

    def __init__(self, m, c, x1, x2):
        self.m = m
        self.c = c
        self.x1 = x1
        self.x2 = x2


def variance(loss):
    sum = 0.0
    for v in loss:
        sum += v
    avg = sum / loss.__len__()
    sum = 0.0
    for v in loss:
        sum += (v - avg) ** 2
    return sum / loss.__len__()


def avg(s):
    sum = 0.0
    for v in s:
        sum += v
    return sum / s.__len__()


def show_train(train_data, train_label, block):
    fig, ax = plt.subplots()
    for i, v in enumerate(train_data):
        if train_label[i][0] == 1.0:
            ax.scatter(v[0], v[1], c='red', marker='^', alpha=1, edgecolors='none')
        else:
            ax.scatter(v[0], v[1], c='blue', marker='o', alpha=1, edgecolors='none')
    ax.legend()
    ax.grid(True)
    plt.show(block=block)


def show_pred(test_data, test_label, pred_label):
    plots = []
    loss = []
    c = 0
    for i, v in enumerate(test_label):
        loss.append(float(pred_label[i]) - float(v))
        if float(v) == pred_label[i]:
            if v == 0:
                plots.append(Result('^', 'red', test_data[i][0], test_data[i][1]))
            else:
                plots.append(Result('o', 'blue', test_data[i][0], test_data[i][1]))
        else:
            c += 1
            if v == 0:
                plots.append(Result('^', 'black', test_data[i][0], test_data[i][1]))
            else:
                plots.append(Result('o', 'black', test_data[i][0], test_data[i][1]))

    print('diff: ', c, ' variance:' + str(variance(loss)))
    fig, ax = plt.subplots()
    for v in plots:
        ax.scatter(v.x1, v.x2, c=v.c, marker=v.m, alpha=1, edgecolors='none')
    ax.legend()
    ax.grid(True)
    plt.show(block=True)


def diff_and_variance(pred, test_label):
    loss = []
    c = 0
    for i, v in enumerate(test_label):
        loss.append(float(pred[i]) - float(v[0]))
        if float(v[0]) != pred[i]:
            c += 1
    var = variance(loss)
    # print('diff: ', c, ' variance: ', str(var))
    return c, var