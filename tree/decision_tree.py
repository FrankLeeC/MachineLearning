# -*- coding:utf-8 -*-

import numpy as np

def get_label_count(x):
    '''
    return:
    unique_labels: [label1, label2, label3] 去重后的样本类别
    label_counts： [3, 4, 8]   每个类别的个数
    count: 样本总数
    '''
    x = np.asarray(x)
    count, feature_counts = np.shape(x)
    unique_labels = np.unique(x[0:count, feature_counts-1:feature_counts])  # [label1, label2, label3]
    label_counts = np.zeros(len(unique_labels))
    for e in x:
        idx = np.argwhere(unique_labels == e[feature_counts-1])
        label_counts[idx[0,0]] += 1
    return unique_labels, label_counts, count


def empirical_entropy(count, label_count):
    '''
    label_count: 有几种样本类别
    count: 样本个数
    return:
    H(D) = -sum_k_K((|Ck|/|D|)*ln(|Ck|/|D|)) 经验熵
    '''
    p = label_count / count
    p_log = np.log(p)
    return np.sum(-p*p_log)

class Node:

    def __init__(self, dataset):
        self.node_type = 'leaf'
        self.dataset = np.asarray(dataset)
        self.left = None
        self.right = None
        self.m = {}
        self.label = None
        self.threshold = None
        self.category = None
        self.judge_feature_idx = None
        self.alpha = 0.5
        self.parent = None
        self.layer = 0

    def set_id(self, id):
        self.id = id

    def get_id(self):
        return self.id

    def set_layer(self, layer):
        '''
        设置当前节点的层数，从1开始
        '''
        self.layer = layer

    def get_layer(self):
        return self.layer

    def get_depth(self):
        if self.node_type == 'leaf':
            return self.layer
        max_depth = 0
        if self.left:
            max_depth = max(self.left.get_depth(), max_depth)
        if self.right:
            max_depth = max(self.right.get_depth(), max_depth)
        if self.m:
            for _, e in self.m.items():
                max_depth = max(e.get_depth(), max_depth)
        return max_depth

    def __set_label(self, label):
        self.label = label

    def walk(self):
        if self.node_type == 'leaf':
            # print('id:%s, parent:%s, %s, label:%s' % (self.id, self.parent.get_id(), 'leaf', self.label))
            print('layer:%d  %s' % (self.layer, 'leaf'))
        else:
            parent_id = None
            if self.parent:
                parent_id = self.parent.get_id()
            if self.split_type == 'continous':
                # print('id:%s, parent:%s, judge_feature_idx:%d, %s, threshold:%f' % (self.id, parent_id, self.judge_feature_idx, 'internel', self.threshold))
                print('layer:%d  %s  split_type:%s' % (self.layer, 'internel', self.split_type))
            else:
                # print('id:%s, parent:%s, judge_feature_idx:%d, %s, category:%s' % (self.id, parent_id, self.judge_feature_idx, 'internel', self.category))
                print('layer:%d  %s  split_type:%s' % (self.layer, 'internel', self.split_type))
            if self.left:
                self.left.walk()
            if self.right:
                self.right.walk()
            if self.m:
                for _, v in self.m.items():
                    v.walk()

    def get_label(self):
        '''
        预测时使用，获取当前叶结点的类别
        '''
        if self.node_type == 'leaf':
            return self.label
        return 'UNKNOWN_CATEGORY'  # 遇到了训练时未出现的特征类别值

    def __set_parent(self, p):
        self.parent = p

    def judge(self, sample):
        if self.node_type == 'internel':
            x = sample[self.judge_feature_idx]
            if self.split_type == 'continous':
                if float(x) <= self.threshold:
                    return self.left
                else:
                    return self.right
            else:
                # print(self.split_type, self.judge_feature_idx, self.m.keys())
                if x in self.m.keys():
                    return self.m[x]
                return None
        else:
            return None

    def get_dataset(self):
        return self.dataset
    
    def clear_dataset(self):
        self.dataset = None
        if self.left:
            self.left.clear_dataset()
        if self.right:
            self.right.clear_dataset()
        if self.m:
            for _, v in self.m:
                v.clear_dataset()

    def split(self, feature_idx, split_type, split_value=None):
        self.node_type = 'internel'
        self.judge_feature_idx = feature_idx
        self.split_type = split_type
        if self.split_type == 'continous':
            self.m = None
        else:
            self.left = None
            self.right = None
        if self.split_type == 'category':  # 类别
            tmp = {}
            for e in self.dataset:
                e = np.asarray(e)
                l = tmp.get(e[feature_idx], list())
                l.append(e)
                tmp[e[feature_idx]] = l
            ids = 0
            for c in tmp.keys():
                ids += 1
                n = Node(np.array(tmp[c]))
                n.category = c
                n.__set_parent(self)
                n.set_id(('%s_%d') % (self.id, ids))
                n.set_layer(self.layer+1)
                self.m[c] = n 
        else:  # 数值型
            d1, d2 = [], []
            self.threshold = split_value
            for e in self.dataset:
                e = np.asarray(e)
                if float(e[feature_idx]) <= split_value:
                    d1.append(e)
                else:
                    d2.append(e)
            l = Node(np.array(d1))
            l.set_id(('%s_%d') % (self.id, 1))
            l.__set_parent(self)
            l.set_layer(self.layer +1)
            r = Node(np.array(d2))
            r.set_id(('%s_%d') % (self.id, 2))
            r.set_layer(self.layer+1)
            r.__set_parent(self)
            self.left = l 
            self.right = r 

    def get_children(self):
        '''
        获取该结点的孩子结点
        '''
        if self.split_type == 'continous':
            return [self.left, self.right]
        return self.m.values()

    def cal_label(self):
        if self.node_type == 'internel':
            for n in self.get_children():
                n.cal_label()
        else:
            m = {}
            _, feature_count = np.shape(self.dataset)
            for e in self.dataset:
                c = m.get(e[feature_count - 1], 0)
                c += 1
                m[e[feature_count - 1]] = c
            new_map = dict(zip(m.values(), m.keys()))
            self.__set_label(new_map[max(new_map.keys())])

    # ---------------- cut ----------------
    def has_child(self):
        if self.node_type == 'leaf':
            return False
        return True

    # def get_bottom_internel(self):
    #     '''
    #     只有 internel node 才能执行该方法   每次由 root 启用
    #     获取最后一层 internel node
    #     '''
    #     l = list()
    #     children = self.get_children()
    #     is_bottom = True
    #     for c in children:
    #         if c.has_child():
    #             is_bottom = False
    #             break
    #     if is_bottom:  # 返回自己
    #         return [self]
    #     if self.split_type == 'continous':
    #         if self.left.node_type == 'internel':
    #             l.extend(self.left.get_bottom_internel())
    #         if self.right.node_type == 'internel':
    #             l.extend(self.right.get_bottom_internel())
    #     else:
    #         for _, v in self.m.items():
    #             if v.node_type == 'internel':
    #                 l.extend(v.get_bottom_internel())
    #     return l

    def retrieve_leaf(self):
        '''
        递归地获取当前结点下的所有叶子结点
        '''
        n = list()
        if self.node_type == 'internel':
            if self.split_type == 'continous':
                ln = self.left.retrieve_leaf()
                if ln:
                    n.extend(ln)
                rn = self.right.retrieve_leaf()
                if rn:
                    n.extend(rn)
                return n
            else:
                for _, v in self.m.items():
                    vn = v.retrieve_leaf()
                    if vn:
                        n.extend(vn)
                return n
        else:
            return [self]

    def cal_entropy(self):
        '''
        计算当前叶子结点的熵
        -sum_1^K(Ntk * log(Ntk/Nt))   Nt表示当前叶子结点的样本个数
        '''
        if self.node_type == 'leaf':
            _, label_count, count = get_label_count(self.dataset)
            p_log = np.log(label_count/count)
            return np.sum(-label_count*p_log)
    
    def get_ct1(self, leaves):
        '''
        sum_1_T(-sum_1^K(Ntk * log(Ntk/Nt)))
        '''
        ct1 = 0.0
        for leaf in leaves:
            ct1 += leaf.cal_entropy()
        return ct1 + self.alpha * len(leaves)

    def get_ct2(self, leaves):
        '''
        sum_1_T(-sum_1^K(Ntk * log(Ntk/Nt)))
        '''
        l = None
        for leaf in leaves:
            if l is None:
                l = leaf.get_dataset()
            else:
                l = np.vstack((l, leaf.get_dataset()))
        _, label_count, count = get_label_count(self.dataset)
        p_log = np.log(label_count/count)
        return np.asarray(l), np.sum(-label_count*p_log) + self.alpha * 1

    def cut(self):
        '''
        计算当前结点所有叶节点的损失函数 ct1，以及剪枝后的损失函数 ct2
        如果 ct2 <= ct1 则剪枝
        否则不变
        '''
        leaves = self.retrieve_leaf()
        ct1 = self.get_ct1(leaves)
        new_dataset, ct2 = self.get_ct2(leaves)
        print(ct1, ct2)
        if ct2 <= ct1:
            self.node_type = 'leaf'
            self.dataset = new_dataset
            self.left = None
            self.right = None
            self.m = None
            self.cal_label()
            return True
        return False

    def retrieve_layer(self, i):
        '''
        获取第 i 层的非叶子结点
        '''
        if self.layer == i and self.node_type == 'internel':
            return [self]
        l = list()
        if self.left:
            left_list = self.left.retrieve_layer(i)
            if left_list:
                l.extend(left_list)
        if self.right:
            right_list = self.right.retrieve_layer(i)
            if right_list:
                l.extend(right_list)
        if self.m:
            for _, e in self.m.items():
                e_list = e.retrieve_layer(i)
                if e_list:
                    l.extend(e_list)
        return l


class DecisionTree:
    '''
    an implementation of C4.5 decision tree
    '''

    def __init__(self, categorical_feature):
        '''
        categorical_feature: nominal feature list 类别特征数组，用特征维度索引标识，从 0 开始
        '''
        if categorical_feature:
            self.categorical_feature = set(categorical_feature)
        self.used_categorical_feature = list()

    def fit(self, x, y):
        '''
        x: count x feature_count   array/matrix
        y: count x 1 array/matrix
        '''
        self.__prepare(x, y)
        self.root = Node(self.training_set)
        self.root.set_id(1)
        self.root.set_layer(1)
        self.__train(self.root)
        self.root.cal_label()
        # self.root.walk()
        self.__cut(self.root)

    def predict(self, x):
        '''
        x: 1 x feature_count  array/matrix
        '''
        t = self.root
        rs = None
        while True:
            tmp = t.judge(x)
            if tmp:
                t = tmp
                continue
            else:
                if t.node_type == 'leaf':
                    rs = t.get_label()
                else:
                    # 分类时遇到未知的特征值
                    # 假如根据特征 A 分为两个结点。左孩子在特征 B 上的特征值有 3 类， 
                    # 右孩子有 4 类，
                    # 此时左孩子缺少的一个值会导致无法继续分流预测 
                    rs = 'UNKNOWN_TYPE' 
                break
        return rs

    def __train(self, node):
        max_gain_rate, feature_idx, split_point = self.__choose_feature(node.get_dataset())
        if (not max_gain_rate and not feature_idx and not split_point) or max_gain_rate <= 1e-5:
            return
        split_type = 'category'
        if split_point:
            split_type = 'continous'
        node.split(feature_idx, split_type, split_point)
        for e in node.get_children():
            self.__train(e)

    def __cut(self, root):
        # cut_count = 0
        # bottom_internel = self.root.get_bottom_internel()
        # for n in bottom_internel:
        #     if n.cut():
        #         cut_count += 1
        # if cut_count > 0:
        #     print(cut_count)
        #     self.__cut(self.root)
        # self.root.walk()
        print('----------------------------------------- cut -------------------------------')
        while True:
            cut = False
            max_depth = self.root.get_depth()
            for i in range(max_depth, 1, -1):
                nodes = self.root.retrieve_layer(i)
                goto = False
                for e in nodes:
                    if e.cut():
                        print('cut---')
                        cut = True
                        goto = True
                        break
                if goto:
                    break
            if not cut:
                break
        # self.root.walk()
        
    def __prepare(self, x, y):
        self.training_set = np.hstack((x, np.asmatrix(y).T))
        self.count, self.feature_counts = np.shape(self.training_set)

    def __sort(self, x, idx, kind='quicksort'):
        if kind == 'quicksort':
            return self.__quick_sort(x, idx)
        raise NotImplementedError

    def __quick_sort(self, x, idx):
        count, _ = np.shape(x)
        if count <= 1:
            return
        i = 0
        j = count - 1
        p = 0
        while i <= j:
            while j >= 0:
                if x[j][idx] < x[p][idx]:
                    x[[p,j], :] =x[[j,p], :]
                    p = j
                    j -= 1
                    break
                j -= 1
            while i <= j:
                if x[i][idx] > x[p][idx]:
                    x[[p,i], :] =x[[i,p], :]
                    p = i
                    i += 1
                    break
                i += 1
        self.__quick_sort(x[0:p,:], idx)
        self.__quick_sort(x[p+1:,:], idx)

    def __get_feature_count(self, x, feature_idx):
        '''
        只有在 categorical feature 时可用，连续性特征不要使用
        return:
        unique_feature: [A, B, C] 第 feature_idx 个特征去重后的集合
        feature_count: [3, 4, 8] 每个特征值的个数
        count: 样本数
        '''
        x = np.asarray(x)
        count, _ = np.shape(x)
        unique_feature = np.unique(x[0:count,feature_idx:feature_idx+1])  # [A, B, C]
        feature_count = np.zeros(len(unique_feature))
        for e in x:
            idx = np.argwhere(unique_feature == e[feature_idx])
            feature_count[idx[0,0]] += 1
        return unique_feature, feature_count, count

    def __empirical_conditional_entropy(self, x, feature_idx):
        '''
        return:
        H(D|A) = sum_i^n((|Di|/|D|)*H(Di)) 条件经验熵
        H(D|A) = -sum_i^n((|Di|/|D|)*sum_k^K((|Dik|/|Di|)*ln(|Dik|/|Di|))) 条件经验熵

        如果是类别型维度，第二个返回值是 None
        如果是连续值维度，第二个返回值表示分割点
        '''
        x = np.asarray(x)
        if self.categorical_feature and feature_idx in self.categorical_feature:  # nominal categorical feature
            unique_features, feature_count, count = self.__get_feature_count(x, feature_idx)
            p_feature = np.array(feature_count) / count
            groups = list()  # 根据 feature 的值分组
            for _ in range(len(unique_features)):  # 初始化 group 
                groups.append([])
            for e in x:
                idx = np.argwhere(unique_features == e[feature_idx])
                groups[idx[0][0]].append(e)
            for i, d in enumerate(groups):
                _, label_count, c = get_label_count(d)
                p_feature[i] *= empirical_entropy(c, label_count)
            return np.sum(p_feature), None
        else:
            m = {}
            self.__sort(x, feature_idx)
            split_point = list()
            count, _ = np.shape(x)
            for i in range(count):
                if float(x[i][feature_idx]) not in split_point:
                    split_point.append(float(x[i][feature_idx]))
            for i, e in enumerate(split_point):
                if i == len(split_point) - 1:
                    break
                group1 = list()
                group2 = list()
                for sample in x:
                    if float(sample[feature_idx]) <= e:
                        group1.append(sample)
                    else:
                        group2.append(sample)
                _, label_count1, c1 = get_label_count(np.array(group1))
                d1 = empirical_entropy(c1, label_count1)
                _, label_count2, c2 = get_label_count(np.array(group2))
                d2 = empirical_entropy(c2, label_count2)
                m[len(group1)*d1/count + len(group2)*d2/count] = e
            return min(m.keys()), m[min(m.keys())]

    def __feature_entropy(self, x, feature_idx):
        '''
        H_A(D) = -sum_i^n((|Di|/|D|)*log(|Di|/|D|))  n 是特征A取值的个数
        '''
        _, feature_count, count = self.__get_feature_count(x, feature_idx)
        p = feature_count / count
        p_log = np.log(p)
        return np.sum(-p*p_log)

    def __choose_feature(self, x):
        '''
        x 数据集 
        计算 x 中每个维度的信息增益比，并返回三个值：max_gain_rate,feature_idx, split_point(if it's continous feature)
        '''
        count, feature_counts = np.shape(x)
        _, label_counts, _ = get_label_count(x)
        if len(label_counts) == 1:  # 没有类别可分
            return None, None, None
        _empirical_entropy = empirical_entropy(count, label_counts)
        m = {}
        terminal = True
        for i in range(feature_counts - 1):  # 计算当前维度的增益比
            if self.used_categorical_feature and i in self.used_categorical_feature:  # categorical feature 只能用一次
                continue
            terminal = False
            empirical_condition_entropy, continous_value = self.__empirical_conditional_entropy(x, i)
            gain = _empirical_entropy - empirical_condition_entropy
            if self.categorical_feature and i in self.categorical_feature:  # 当前是离散的类别特征，计算增益率
                gain_rate = gain / (self.__feature_entropy(x, i)+1e-8)
                m[gain_rate] = (i, continous_value)
            else:
                m[gain] = (i, continous_value)
        if terminal:
            return None, None, None
        if self.categorical_feature and m[max(m.keys())][0] in self.categorical_feature:
            self.used_categorical_feature.append(m[max(m.keys())][0])
        return max(m.keys()), m[max(m.keys())][0], m[max(m.keys())][1]
