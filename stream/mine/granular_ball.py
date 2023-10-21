# -*- coding: utf-8 -*-
"""
@Time ： 2023/5/13 12:05
@Auth ： daiminggao
@File ：granular_ball.py
@IDE ：PyCharm
@Motto:咕咕咕
"""


class GranularBall:
    def __init__(self, data):
        self.data = data
        self.center = self.data[:, :-1].mean(0)
        self.radius = self.get_radius()

    def get_radius(self):
        if len(self.data) == 1:
            return 0
        return max(((self.data[:, :-1] - self.center) ** 2).sum(axis=1) ** 0.5)
