# -*- coding: utf-8 -*-
"""
Created on Wed May 11 10:35:51 2022

@author: xiejiang
"""

import numpy as np
import matplotlib.pyplot as plt
from stream.mine import FuzzyClustering_no_random


class GB:
    def __init__(self, data, label):
        self.data = data
        self.center = self.data[:, :-1].mean(0)
        self.radius = self.get_radius()
        self.label = label
        self.num = len(data)

    def get_radius(self):
        return max(((self.data[:, :-1] - self.center) ** 2).sum(axis=1) ** 0.5)


def get_density_volume(gb):
    num = len(gb)
    center = gb.mean(0)
    diffMat = np.tile(center, (num, 1)) - gb
    sqDiffMat = diffMat ** 2
    sqDistances = sqDiffMat.sum(axis=1)
    distances = sqDistances ** 0.5
    sum_radius = 0
    if len(distances) == 0:
        print("0")
    for i in distances:
        sum_radius = sum_radius + i
    mean_radius = sum_radius / num
    if mean_radius != 0:
        density_volume = num / sum_radius
    else:
        density_volume = num
    return density_volume


# 无参遍历粒球是否需要分裂，根据子球和父球的比较，不带断裂判断的分裂,1分2
def division_2_2(gb_list):
    gb_list_new = []
    for gb_data in gb_list:
        if len(gb_data) >= 8:
            ball_1, ball_2 = spilt_ball_fuzzy(gb_data)  # 有模糊
            if len(ball_1) == 1 or len(ball_2) == 1:
                gb_list_new.append(gb_data)
                continue
            if len(ball_1) == 0 or len(ball_2) == 0:
                gb_list_new.append(gb_data)
                continue
            parent_dm = get_density_volume(gb_data[:, :-1])
            child_1_dm = get_density_volume(ball_1[:, :-1])
            child_2_dm = get_density_volume(ball_2[:, :-1])
            w1 = len(ball_1) / (len(ball_1) + len(ball_2))
            w2 = len(ball_2) / (len(ball_1) + len(ball_2))
            w_child_dm = (w1 * child_1_dm + w2 * child_2_dm)  # 加权子粒球DM
            t2 = (w_child_dm > parent_dm)  # 加权DM上升
            if t2:
                gb_list_new.extend([ball_1, ball_2])
            else:
                gb_list_new.append(gb_data)
        else:
            gb_list_new.append(gb_data)
    return gb_list_new


# fuzzy粒球分裂
def spilt_ball_fuzzy(data):
    cluster = FuzzyClustering_no_random.FCM_no_random(data[:, :-1], 2)
    ball1 = data[cluster == 0, :]
    ball2 = data[cluster == 1, :]
    return [ball1, ball2]


def get_radius(gb_data):
    # origin get_radius 7*O(n)
    sample_num = len(gb_data)
    center = gb_data.mean(0)
    diffMat = np.tile(center, (sample_num, 1)) - gb_data
    sqDiffMat = diffMat ** 2
    sqDistances = sqDiffMat.sum(axis=1)
    distances = sqDistances ** 0.5
    radius = max(distances)
    return radius


def gb_plot(gb_dict, noise, t):
    color = {
        0: '#707afa',
        1: '#ffe135',
        2: '#16ccd0',
        3: '#ed7231',
        4: '#0081cf',
        5: '#afbed1',
        6: '#bc0227',
        7: '#d4e7bd',
        8: '#f8d7aa',
        9: '#fecf45',
        10: '#f1f1b8',
        11: '#b8f1ed',
        12: '#ef5767',
        13: '#e7bdca',
        14: '#8e7dfa',
        15: '#d9d9fc',
        16: '#2cfa41',
        17: '#e96d29',
        18: '#7f722f',
        19: '#bd57fa',
        20: '#e4f788',
        21: '#fb8e94',
        22: '#b8d38f',
        23: '#e3a04f',
        24: '#edc02f',

        25: '#ff8444',
        26: '#a6dceb',
        27: '#fdd3a2',
        28: '#e6b1c2',
        29: '#9bb7d4',
        30: '#fedb5c',
        31: '#b2e1e0',
        32: '#f8c0b6',
        33: '#c8bfe7',
        34: '#f4af81',
        35: '#a3a3a3',
        36: '#bce784',
        37: '#8d6e63',
        38: '#e9e3c9',
        39: '#f5e9b2',
        40: '#ffba49',
        41: '#c0c0c0',
        42: '#d3a7b5',
        43: '#f2c2e0',
        44: '#b7dd29',
        45: '#dcf7c1',
        46: '#6f9ed7',
        47: '#d8a8c3',
        48: '#76c57f',
        49: '#f6e9cd',
        50: '#a16fd8',
        51: '#c5e6a7',
        52: '#f98f76',
        53: '#b3d6e3',
        54: '#efc8a5',
        55: '#5c9aa1',
        56: '#d3e1b6',
        57: '#a87ac8',
        58: '#e2d095',
        59: '#c95a3b',
        60: '#7fb4d1',
        61: '#f7d28e',
        62: '#b9c9b0',
        63: '#e994b9',
        64: '#8bc9e4',
        65: '#e6b48a',
        66: '#acd4d8',
        67: '#f3e0b0',
        68: '#57a773',
        69: '#d9bb7b',
        70: '#8e73e5',
        71: '#f4c4e3',
        72: '#75a88b',
        73: '#c0d4eb',
        74: '#a46c9b',
        75: '#d7e3a0',
        76: '#bd5f36',
        77: '#77c5b8',
        78: '#e8b7d5',
        79: '#4e8746',
        80: '#f0d695',
        81: '#9b75cc',
        82: '#c2e68a',
        83: '#f56e5c',
        84: '#a9ced0',
        85: '#e18a6d',
        86: '#6291b1',
        87: '#d1dbab',
        88: '#c376c5',
        89: '#8fc9b5',
        90: '#f7e39e',
        91: '#6d96b8',
        92: '#f9c0a6',
        93: '#63a77d',
        94: '#dbb8e9',
        95: '#9aa3d6',
        96: '#e3ca7f',
        97: '#b15d95',
        98: '#88c2e0',
        99: '#f4c995',
        100: '#507c94',
    }
    label_c = {
        0: 'cluster-1',
        1: 'cluster-2',
        2: 'cluster-3',
        3: 'cluster-4',
        4: 'cluster-5',
        5: 'cluster-6',
        6: 'cluster-7',
        7: 'cluster-8',
        8: 'cluster-9',
        9: 'cluster-10',
        10: 'cluster-11',
        11: 'cluster-12',
        12: 'cluster-13',
        13: 'cluster-14',
        14: 'cluster-15',
        15: 'cluster-16',
        16: 'cluster-17',
        17: 'cluster-18',
        18: 'cluster-19',
        19: 'cluster-20',
        20: 'cluster-21',
        21: 'cluster-22',
        22: 'cluster-23',
        23: 'cluster-24',
        24: 'cluster-25',
        25: 'cluster-26',
        26: 'cluster-27',
        27: 'cluster-28',
        28: 'cluster-29',
        29: 'cluster-30',
        30: 'cluster-31',
        31: 'cluster-32',
        32: 'cluster-33',
        33: 'cluster-34',
        34: 'cluster-35',
        35: 'cluster-36',
        36: 'cluster-37',
        37: 'cluster-38',
        38: 'cluster-39',
        39: 'cluster-40',
        40: 'cluster-41',
        41: 'cluster-42',
        42: 'cluster-43',
        43: 'cluster-44',
        44: 'cluster-45',
        45: 'cluster-46',
        46: 'cluster-47',
        47: 'cluster-48',
        48: 'cluster-49',
        49: 'cluster-50',
        50: 'cluster-51',
        51: 'cluster-52',
        52: 'cluster-53',
        53: 'cluster-54',
        54: 'cluster-55',
        55: 'cluster-56',
        56: 'cluster-57',
        57: 'cluster-58',
        58: 'cluster-59',
        59: 'cluster-60',
        60: 'cluster-61',
        61: 'cluster-62',
        62: 'cluster-63',
        63: 'cluster-64',
        64: 'cluster-65',
        65: 'cluster-66',
        66: 'cluster-67',
        67: 'cluster-68',
        68: 'cluster-69',
        69: 'cluster-70',
        70: 'cluster-71',
        71: 'cluster-72',
        72: 'cluster-73',
        73: 'cluster-74',
        74: 'cluster-75',
        75: 'cluster-76',
        76: 'cluster-77',
        77: 'cluster-78',
        78: 'cluster-79',
        79: 'cluster-80',
        80: 'cluster-81',
        81: 'cluster-82',
        82: 'cluster-83',
        83: 'cluster-84',
        84: 'cluster-85',
        85: 'cluster-86',
        86: 'cluster-87',
        87: 'cluster-88',
        88: 'cluster-89',
        89: 'cluster-90',
        90: 'cluster-91',
        91: 'cluster-92',
        92: 'cluster-93',
        93: 'cluster-94',
        94: 'cluster-95',
        95: 'cluster-96',
        96: 'cluster-97',
        97: 'cluster-98',
        98: 'cluster-99',
        99: 'cluster-100',
        100: 'cluster-101',
    }
    cluster_label_list = []
    for i in range(0, len(gb_dict)):
        if gb_dict[i].label not in cluster_label_list:
            cluster_label_list.append(gb_dict[i].label)
    plt.figure(figsize=(15, 15))
    for i in range(0, len(cluster_label_list)):
        if cluster_label_list[i] == -1:
            cluster_label_list.remove(-1)
            break
    cluster = {}
    for label in cluster_label_list:
        for key in gb_dict.keys():
            if gb_dict[key].label == label:
                if label not in cluster.keys():
                    cluster[label] = gb_dict[key].data
                else:
                    cluster[label] = np.append(cluster[label], gb_dict[key].data, axis=0)
    theta = np.arange(0, 2 * np.pi, 0.01)
    for i, key in enumerate(cluster.keys()):
        plt.scatter(cluster[key][:, 0], cluster[key][:, 1], s=4, c=color[i], linewidths=5, alpha=0.9,
                    marker='o', label=label_c[i])
        # 画圆
        for gb in gb_dict.values():
            if gb.label == key:
                center = gb.data[:, :-1].mean(0)
                r = gb.get_radius()
                x = center[0] + r * np.cos(theta)
                y = center[1] + r * np.sin(theta)
                plt.plot(x, y, c=color[i])
    if len(noise) > 0:
        plt.scatter(noise[:, 0], noise[:, 1], s=40, c='black', linewidths=2, alpha=1, marker='x', label='noise')
    for key in gb_dict.keys():
        for i in range(0, len(cluster_label_list)):
            if gb_dict[key].label == -1:
                plt.scatter(gb_dict[key].data[:, 0], gb_dict[key].data[:, 1], s=40, c='black', linewidths=2, alpha=1,
                            marker='x')
    plt.legend(loc=1, fontsize=12)
    plt.title("no." + str(t), size=20, loc='center')
    plt.show()


# 缩小粒球
def minimum_ball(gb_list, radius_detect):
    gb_list_temp = []
    for gb_data in gb_list:
        if len(gb_data) <= 2:
            if (len(gb_data) == 2) and (get_radius(gb_data) > 1.2 * radius_detect):
                gb_list_temp.append(np.array([gb_data[0], ]))
                gb_list_temp.append(np.array([gb_data[1], ]))
            else:
                gb_list_temp.append(gb_data)
        else:
            if get_radius(gb_data) <= 1.2 * radius_detect:
                gb_list_temp.append(gb_data)
            else:
                ball_1, ball_2 = spilt_ball_fuzzy(gb_data)  # 有模糊
                if len(ball_1) == 1 or len(ball_2) == 1:
                    if get_radius(gb_data) > radius_detect:
                        gb_list_temp.extend([ball_1, ball_2])
                    else:
                        gb_list_temp.append(gb_data)
                else:
                    gb_list_temp.extend([ball_1, ball_2])

    return gb_list_temp


# 归一化
def normalized_ball(gb_list, radius_detect):
    gb_list_temp = []
    for gb_data in gb_list:
        if len(gb_data) <= 2:
            if (len(gb_data) == 2) and (get_radius(gb_data) > 1.5 * radius_detect):
                gb_list_temp.append(np.array([gb_data[0], ]))
                gb_list_temp.append(np.array([gb_data[1], ]))
            else:
                gb_list_temp.append(gb_data)
        else:
            if get_radius(gb_data[:, :-1]) <= 1.5 * radius_detect:
                gb_list_temp.append(gb_data)
            else:
                ball_1, ball_2 = spilt_ball_fuzzy(gb_data)  # 有模糊
                gb_list_temp.extend([ball_1, ball_2])
    return gb_list_temp

