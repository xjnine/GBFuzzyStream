# -*- coding: utf-8 -*-
"""
Created on Thu Oct 13 08:51:09 2022

@author: xjnine
"""

import numpy as np


def FCM_no_random(X, c_clusters=2, m=2, eps=10):
    membership_mat = np.random.random((len(X), c_clusters))
    membership_mat = np.divide(membership_mat, np.sum(membership_mat, axis=1)[:, np.newaxis])
    # In the first split, select a fixed center to remove the randomness
    center = X.mean(0)
    n, d = np.shape(X)
    dist_1_mat = np.sqrt(np.sum(np.asarray(center - X) ** 2, axis=1).astype('float'))  # 离中心最远点之间距离矩阵
    index_1_mat = np.where(dist_1_mat == np.max(dist_1_mat))  # 离中心最远点下标矩阵
    if len(X[index_1_mat, :][0]) >= 2:  # 如果存在多个最远点下标
        p1 = np.reshape(X[index_1_mat, :][0][0], [d, ])  # 取第一个最远点
    else:
        p1 = np.reshape(X[index_1_mat, :], [d, ])
    dist_2_mat = np.sqrt(np.sum(np.asarray(p1 - X) ** 2, axis=1).astype('float'))  # 离p1最远点之间距离矩阵
    index_2_mat = np.where(dist_2_mat == np.max(dist_2_mat))  # 离p1最远点下标矩阵
    if len(X[index_2_mat, :][0]) >= 2:
        p2 = np.reshape(X[index_2_mat, :][0][0], [d, ])  # 取第一个最远点
    else:
        p2 = np.reshape(X[index_2_mat, :], [d, ])
    c_p1 = (center + p1) / 2
    c_p2 = (center + p2) / 2

    Centroids = np.array([c_p1, c_p2])
    # n_c_distance_mat每个点到c_p1,c_p2的距离矩阵
    while True:
        n_c_distance_mat = np.zeros((len(X), c_clusters))
        for i, x in enumerate(X):
            for j, c in enumerate(Centroids):
                n_c_distance_mat[i][j] = np.linalg.norm(x - c, 2)
        new_membership_mat = np.zeros((len(X), c_clusters))
        for i, x in enumerate(X):
            for j, c in enumerate(Centroids):
                if 0 in n_c_distance_mat[i]:
                    if n_c_distance_mat[i][0] == 0:
                        new_membership_mat[i][0] = 1
                        new_membership_mat[i][1] = 0
                        break
                    else:
                        new_membership_mat[i][1] = 1
                        new_membership_mat[i][0] = 0
                else:
                    new_membership_mat[i][j] = 1. / np.sum(
                        (n_c_distance_mat[i][j] / (n_c_distance_mat[i])) ** (2 / (m - 1)))
        if np.sum(abs(new_membership_mat - membership_mat)) < eps:
            break
        membership_mat = new_membership_mat
        working_membership_mat = membership_mat ** m
        Centroids = np.divide(np.dot(working_membership_mat.T, X),
                              np.sum(working_membership_mat.T, axis=1)[:, np.newaxis])
    return np.argmax(new_membership_mat, axis=1)
