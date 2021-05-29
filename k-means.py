import matplotlib.pyplot as plt
import random
import collections
import pandas as pd
import os
from mpl_toolkits.mplot3d import Axes3D
from sklearn.cluster import KMeans
from sklearn import datasets, model_selection
import numpy as np


def loadDataSet(file_path):
    dataMat = []  # 列表list
    labelMat = []
    txt = open(file_path)
    count = 1
    for line in txt.readlines():
        lineArr = line.strip().split(',')  # strip():返回一个带前导和尾随空格的字符串的副本,split():默认以空格为分隔符，空字符串从结果中删除
        if count < 8:
            count += 1
            continue
        dataMat.append([float(lineArr[0]), float(lineArr[1])])
        if float(lineArr[2]) == -1:
            labelMat.append(-1)
        else:
            labelMat.append(1)
    x_train = dataMat[:100]
    y_train = labelMat[:100]
    x_test = dataMat[:100]
    y_test = labelMat[:100]
    return x_train, y_train, x_test, y_test


def points_avg(points, dim):
    center = []
    d = dim
    for i in range(d):
        d_s = 0
        for p in points:
            d_s += p[i]
        center.append(d_s / float(len(points)))
    return center


def plot_class(data, label):
    x_train, y_train = np.array(data), np.array(label)
    x_min, x_max = x_train[:, 0].min() - .5, x_train[:, 0].max() + .5
    y_min, y_max = x_train[:, 1].min() - .5, x_train[:, 1].max() + .5

    plt.figure(1, figsize=(8, 6))
    plt.clf()
    map_color = {-1: 'r', 1: 'g'}
    color = list(map(lambda x: map_color[x], y_train))
    # Plot the training points

    plt.scatter(x_train[:, 0], x_train[:, 1], c=color, marker='o',
                edgecolor='k')
    plt.xlabel('Banana length')
    plt.ylabel('Banana width')

    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)
    plt.xticks(())
    plt.yticks(())
    plt.show()


# x_train, y_train, _, _ = loadDataSet(file_path0)
def create_data(num):
    data = collections.defaultdict(list)
    li = [[] * 4] * num
    li[0] = [5, 18, 10, 25]
    li[1] = [23, 41, 5, 19]
    li[2] = [30, 44, 27, 42]
    j = 0
    for l in li:
        name = 'class_' + str(j)
        j += 1
        for i in range(50):
            x1 = random.uniform(l[0], l[1])
            y1 = random.uniform(l[2], l[3])
            data[name].append(((round(x1, 2), round(y1, 2))))
    return data


def k_class_show(train_data, k=None):
    if train_data:
        pass
    else:
        train_data = create_data(3)
        np.save('k-means.npy', train_data)
    plt.figure(2, figsize=(8, 6))
    plt.clf()
    x1 = np.array(train_data['class_0'])
    x2 = np.array(train_data['class_1'])
    x3 = np.array(train_data['class_2'])
    plt.scatter(x1[:, 0], x1[:, 1], c='r', marker='o')
    plt.scatter(x2[:, 0], x2[:, 1], c='b', marker='o')
    plt.scatter(x3[:, 0], x3[:, 1], c='g', marker='o')
    # plt.scatter(k[:, 0], k[:, 1], s=150, marker='v')
    plt.xlim(0, 50)
    plt.ylim(0, 50)
    plt.show()


data = np.load('./data/k-means/k-means.npy').item()  # 注意这里dict的load加上了item()
centers = []
for i in range(3):
    p = data["class_%d" % i]
    centers.append(points_avg(p, 2))
print(centers)

# data_set = data['class_0'] + data['class_1'] + data['class_2']
# ks = KMeans(n_clusters=3, random_state=10)
# ks.fit(data_set)
# print(ks.cluster_centers_)
# t = ks.cluster_centers_
# k_class_show(data, k=t)
