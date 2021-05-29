from collections import defaultdict
import random
from math import sqrt
import numpy as np


def point_avg(points,dim):
    """
    Accepts a list of points, each with the same number of dimensions.
    NB. points can have more dimensions than 2

    Returns a new point which is the center of all the points.
    """
    new_center = []
    d = dim
    for i in range(d):
        d_sum = 0
        for p in points:
            d_sum += p[i]
        new_center.append(d_sum / float(len(points)))
    return new_center


def update_centers(data_set, assignments):
    """
    Accepts a dataset and a list of assignments; the indexes
    of both lists correspond to each other.
    Compute the center for each of the assigned groups.
    Return `k` centers where `k` is the number of unique assignments.
    """
    new_means = defaultdict(list)
    centers = []
    for assignment, point in zip(assignments, data_set):
        new_means[assignment].append(point)

    for _, points in new_means.items():
        centers.append(point_avg(points,2))

    return centers


def assign_points(data_points, centers):
    """
    Given a data set and a list of points betweeen other points,
    assign each point to an index that corresponds to the index
    of the center point on it's proximity to that point.
    Return a an array of indexes of centers that correspond to
    an index in the data set; that is, if there are N points
    in `data_set` the list we return will have N elements. Also
    If there are Y points in `centers` there will be Y unique
    possible values within the returned list.
    """
    assignments = []
    for point in data_points:
        shortest = 200  # positive infinity
        shortest_index = 0
        for i in range(len(centers)):
            val = distance(point, centers[i])
            if val < shortest:
                shortest = val
                shortest_index = i
        assignments.append(shortest_index)
    return assignments


def distance(a, b):
    """
    """
    d = len(a)
    dis_sum = 0

    for i in range(d):
        dis_sum += (a[i] - b[i]) ** 2
    return sqrt(dis_sum)


def generate_k(data_set, k):
    """
    Given `data_set`, which is an array of arrays,
    find the minimum and maximum for each coordinate, a range.
    Generate `k` random points between the ranges.
    Return an array of the random points within the ranges.
    """
    # centers = []
    # dimensions = len(data_set[0])
    # min_max = defaultdict(int)
    #
    # for point in data_set:
    #     for i in range(dimensions):
    #         val = point[i]
    #         min_key = 'min_%d' % i
    #         max_key = 'max_%d' % i
    #         if min_key not in min_max or val < min_max[min_key]:
    #             min_max[min_key] = val
    #         if max_key not in min_max or val > min_max[max_key]:
    #             min_max[max_key] = val
    #
    # for _k in range(k):
    #     rand_point = []
    #     for i in range(dimensions):
    #         min_val = min_max['min_%d' % i]
    #         max_val = min_max['max_%d' % i]
    #
    #         rand_point.append(uniform(min_val, max_val))
    #
    #     centers.append(rand_point)
    #
    # return centers
    centers = []
    n = len(data_set)
    for i in range(k):
        s = random.randint(0, n)
        centers.append(data_set[s])
    return centers


def k_means(dataset, k):
    k_points = generate_k(dataset, k)
    assignments = assign_points(dataset, k_points)
    old_assignments = None
    while assignments != old_assignments:
        new_centers = update_centers(dataset, assignments)
        old_assignments = assignments
        assignments = assign_points(dataset, new_centers)
    return new_centers


train_data = np.load('./data/k-means/k-means.npy').item()
data = train_data['class_0'] + train_data['class_1'] + train_data['class_2']
data = np.array(data)
print(k_means(data,3))