# -*-encoding=utf-8-*-
import codecs


def read(path, spt=','):
    x = []
    y = []
    with codecs.open(path) as file:
        while True:
            line = file.readline().strip()
            if line:
                d = line.strip().split(spt)
                m = []
                dim = d.__len__()
                for i in range(dim - 1):
                    m.append(float(d[i]))
                x.append(m)
                y.append([float(d[dim - 1])])
            else:
                break
    return x, y


def write(path, data, label):
    with codecs.open(path, mode='w', encoding='utf-8') as file:
        for i, v in enumerate(data):
            s = ''.join([(str(_v) + ',') for j, _v in enumerate(v)])
            file.write(s + str(label[i][0]) + '\n')
