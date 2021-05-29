import matplotlib.pyplot as plt
import random
import collections
import numpy as np
from math import sqrt


def point_avg(points, dimensions):
    new_center = []
    d = dimensions
    for i in range(d):
        d_sum = 0
        for p in points:
            d_sum += p[i]
        new_center.append(d_sum / float(len(points)))
    return new_center


def update_centers(data_set, assign):
    new_means = collections.defaultdict(list)
    centers = []
    for ass, point in zip(assign, data_set):
        new_means[ass].append(point)
    for _, points in new_means.items():
        centers.append(point_avg(points, 2))
    return centers


def distance(a, b):
    d = len(a)
    dis_sum = 0

    for i in range(d):
        dis_sum += (a[i] - b[i]) ** 2
    return sqrt(dis_sum)


def assign_points(data_points, centers):  # 可以尝试使用assignment这种方式衡量循环是否中止
    assignments = []
    for point in data_points:
        shortest = 200
        shortestidx = 0
        for i in range(len(centers)):
            v = distance(centers[i], point)
            if v < shortest:
                shortest = v
                shortestidx = i
        assignments.append(shortestidx)
    return assignments


def generate_k(data_set, k):
    # centers = []
    # d = len(data_set[0])
    # min_max = collections.defaultdict(int)
    #
    # for point in data_set:
    #     for i in range(d):
    #         v = point[i]
    #         mi = 'min_%d' % i
    #         mx = 'max_%d' % i
    #         if mi not in min_max or v < min_max[mi]:
    #             min_max[mi] = v
    #         if mx not in min_max or v > min_max[mx]:
    #             min_max[mx] = v
    #
    # for j in range(k):
    #     r_c=[]
    #     for i in range(d):
    #         min_v=min_max['min_%d'%i]
    #         max_v=min_max['max_%d'%i]
    #         r_c.append(round(random.uniform(min_v,max_v),2))
    #     centers.append(r_c)

    centers = []
    n = len(data_set)
    for i in range(k):
        s = random.randint(0, n)
        centers.append(data_set[s])
    return centers


def k_means(dataset, num, c=True):
    # k_points = generate_k(data_set=dataset, k=num)
    # assignments = assign_points(dataset, k_points)
    # new_centers = []
    # oldu = None
    # old_centers = k_points
    # if c:
    #     while assignments != oldu:
    #         new_centers = updata_centers(dataset, assignments)
    #         oldu = assignments
    #         assignments = assign_points(dataset, new_centers)
    #     return assignments
    # else:
    #     while new_centers != old_centers:
    #         new_centers = updata_centers(dataset, assignments)
    #         old_centers = new_centers
    #         assignments = assign_points(dataset, new_centers)
    #     return assignments
    k_points = generate_k(dataset, num)
    assignments = assign_points(dataset, k_points)
    old_assignments = None
    while assignments != old_assignments:
        new_centers = update_centers(dataset, assignments)
        old_assignments = assignments
        assignments = assign_points(dataset, new_centers)
    return new_centers, assignments


def DBI(c, data):
    c0, c1, c2 = c[0], c[1], c[2]
    avg = collections.Counter(s)
    dcen = {"0-1": 0, "1-2": 0, "0-2": 0}
    dcen["0-1"], dcen['1-2'], dcen["0-2"] = distance(c0, c1), distance(c1, c2), distance(c0, c2)

    k = 0
    for i in range(len(c)):
        d_s = 0
        for p in data[k:k + 50]:
            d_s += distance(p, c[i])
        avg[i] = d_s / 50.0
    DBI = ((avg[0] + avg[1]) / dcen["0-1"] + (avg[1] + avg[2]) / dcen["1-2"] + (avg[1] + avg[2]) / dcen["1-2"]) / 3.0
    return DBI


def main():
    train_data = np.load('./data/k-means/k-means.npy').item()
    data = train_data['class_0'] + train_data['class_1'] + train_data['class_2']
    centers, assignments = k_means(dataset=data, num=3, c=False)
    return centers, assignments, data

def move(N):
    a = list()
    s=N
    while s > 0:
        t = s % 10
        s = s // 10
        a.append(t)
    a = a[::-1]
    n = len(a)
    if n == 0:
        return 0
    for i in range(1, n):
        if a[i - 1] > a[i]:
            return a[i - 1] * 10 ** (n - i) - 1
    return N

if __name__ == "__main__":
    c, s, data = main()
    print("Centers:", [[round(x[0], 2), round(x[1], 2)] for x in c])
    dbi = DBI(c, data)
    print("DBI:", round(dbi,3))




