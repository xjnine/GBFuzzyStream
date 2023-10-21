import numpy as np

from stream.mine.HyperballClustering import spilt_ball_fuzzy


class MicroBall:
    def __init__(self, data, label):
        self.data = data
        self.center = self.get_center()
        self.radius = self.get_radius()
        self.label = label
        self.num = self.get_num()
        self.DM = self.get_dm()
        # (zt)加权权重
        self.weight_info = {}
        self.weight = -1
        self.data_num_in_sometime = {}

    def get_radius(self):
        if len(self.data) == 1:
            return 0
        return max(((self.data[:, :-1] - self.center) ** 2).sum(axis=1) ** 0.5)

    def get_center(self):
        return self.data[:, :-1].mean(0)

    def get_num(self):
        return len(self.data)

    def get_dm(self):
        diffMat = np.tile(self.center, (self.num, 1)) - self.data[:, :-1]
        sqDiffMat = diffMat ** 2
        sqDistances = sqDiffMat.sum(axis=1)
        distances = sqDistances ** 0.5
        sum_dis = 0
        for i in distances:
            sum_dis += i
        DM = sum_dis / self.num
        return DM

    def init_weight(self, t, new_data_num):
        self.weight = 1
        self.data_num_in_sometime[t] = new_data_num  # 记录该时刻对应数据量

    def add_weight_info(self, t, new_data_num):
        if t in self.data_num_in_sometime.keys():
            self.data_num_in_sometime[t] += new_data_num
        else:
            self.data_num_in_sometime[t] = new_data_num  # 记录该时刻对应数据量

    def update_weight(self, t, lam):
        data_num = self.num
        current_weight = 0
        for key in self.data_num_in_sometime.keys():
            current_weight += (self.data_num_in_sometime[key] / data_num) * (2.0 ** ((-lam) * (t - key)))
        self.weight = current_weight

    def spilt_ball(self):
        data = self.data
        if len(np.unique(data, axis=0)) == 1:
            return False
        data1 = []
        data2 = []
        n, m = data.shape
        X = data.T
        G = np.dot(X.T, X)
        H = np.tile(np.diag(G), (n, 1))
        D = np.sqrt(np.abs(H + H.T - G * 2))  # 计算欧式距离
        r, c = np.where(D == np.max(D))  # 查找距离最远的两个点坐标
        r1 = r[1]
        c1 = c[1]
        if r1 == c1:
            return False
        for j in range(0, len(data)):  # 对于距离最远的两个点之外的点，根据到两点距离不同划分到不同的簇中
            if D[j, r1] < D[j, c1]:
                data1.extend([data[j, :]])
            else:
                data2.extend([data[j, :]])
        if len(data1) * len(data2) == 0:
            return False
        else:
            data1 = np.array(data1)
            data2 = np.array(data2)
            return [MicroBall(data=data1, label=None),
                    MicroBall(data=data2, label=None)]

    def is_division(self):
        if self.num >= 8:
            if len(np.unique(self.data[:, :-1])) == 1:
                spilt_result = False
            else:
                spilt_result = spilt_ball_fuzzy(self.data)
                if len(spilt_result[0]) * len(spilt_result[1]) == 1:
                    return False
                if len(spilt_result[0]) * len(spilt_result[1]) == 0:
                    spilt_result = False
            if spilt_result:
                ball_1 = MicroBall(data=spilt_result[0], label=None)
                ball_2 = MicroBall(data=spilt_result[1], label=None)
                DM_parent = self.DM
                DM_child_1 = ball_1.DM * (ball_1.num / self.num)
                DM_child_2 = ball_2.DM * (ball_2.num / self.num)
                weight_child_DM = DM_child_2 + DM_child_1
                t1 = weight_child_DM < DM_parent
                if t1:
                    # （zt）继承母球权重
                    time_index1 = ball_1.data[:, -1]
                    time_index2 = ball_2.data[:, -1]
                    for key in self.data_num_in_sometime.keys():
                        if key in ball_1.data_num_in_sometime.keys() and key in ball_2.data_num_in_sometime.keys():
                            continue
                        if key in time_index1:
                            ball_1.data_num_in_sometime[key] = np.sum(time_index1 == key)
                        if key in time_index2:
                            ball_2.data_num_in_sometime[key] = np.sum(time_index2 == key)
                    return [ball_1, ball_2]
                else:
                    return False
            else:
                return False
        return False

    def insert_ball(self, gb, t):
        self.data = np.append(self.data, gb.data, axis=0)
        self.center = self.get_center()
        self.radius = self.get_radius()
        self.num = self.get_num()
        self.DM = self.get_dm()
        self.add_weight_info(t, len(gb.data))
        return self.is_division()

