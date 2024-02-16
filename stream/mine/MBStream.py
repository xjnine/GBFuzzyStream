import math
import sys
from sklearn.decomposition import PCA
from stream.mine.HyperballClustering import *
from stream.mine.MicroCluster import MicroBall
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from stream.mine.DPC import get_cluster_DPC
from stream.mine.granular_ball import GranularBall


class MBStream:
    def __init__(self, data, dataset, plot_evaluate_flag, lam=1):
        self.datasetName = dataset
        self.timeSeries = data.values[:, -1]
        self.timeSpan = 1
        self.data = self.normalized(data)
        self.begin = self.timeSeries[0]
        self.stop = self.timeSeries[-1]
        self.timeIndex = 0
        self.lam = lam
        self.threshold = 0.8  # 0.3
        self.trueLabel = list(map(str, data.values[:, -2]))
        self.micro_balls = self.init_v1(plot_evaluate_flag)

    @staticmethod
    def normalized(data):
        min_max_scaler = MinMaxScaler(feature_range=(0, 1))  # 数据缩放
        value = data.values[:, :-2]
        time_index = data.values[:, -1]
        if len(value[0]) > 2:
            pca = PCA(n_components=2)
            value = pca.fit_transform(value)
            value = min_max_scaler.fit_transform(value)
        else:
            value = min_max_scaler.fit_transform(value)
        time_index = time_index.reshape(len(time_index), 1)
        data = np.append(value, time_index, axis=1)  # 拼接时间戳到数据上（data，index）
        return data

    # v1.0
    def init_v1(self, plot_evaluate_flag):
        init_data = []
        # 收集第一时刻数据放入init_data
        for x in range(0, len(self.data)):
            if self.timeSeries[x] <= self.begin:
                init_data.append(self.data[x])
            else:
                self.timeIndex = x
                break
        clusters, gb_list, gb_dict = self.connect_ball_DPC(1,np.array(init_data))
        init_mb_list = []
        for data in gb_list:
            mb = MicroBall(data, None)
            mb.init_weight(self.begin, len(data))
            init_mb_list.append(mb)
        if plot_evaluate_flag:
            gb_plot(gb_dict, [], 1)
        print("now have", len(gb_list), "micro-balls")
        return init_mb_list

    @staticmethod
    def get_nearest_micro_ball(sample, micro_balls):
        smallest_distance = sys.float_info.max
        nearest_micro_ball = None
        nearest_micro_ball_index = -1
        for i, micro_ball in enumerate(micro_balls):
            current_distance = np.linalg.norm(micro_ball.center - sample)
            if current_distance < smallest_distance:
                smallest_distance = current_distance
                nearest_micro_ball = micro_ball
                nearest_micro_ball_index = i
        if nearest_micro_ball is None:
            print("nearest_micro_ball is None")
        return nearest_micro_ball_index, nearest_micro_ball, smallest_distance

    def fit_predict(self, plot_evaluate_flag):
        t = self.begin + self.timeSpan
        while t <= self.stop:
            new_samples = []
            for i in range(self.timeIndex, len(self.timeSeries)):
                if math.isclose(self.timeSeries[i], t):
                    new_samples.append(self.data[i])
                    if i == len(self.timeSeries) - 1:
                        break
                    if not math.isclose(self.timeSeries[i + 1], t):
                        self.timeIndex = i + 1
                        break
            if len(new_samples) == 0:
                print(f"{t} have no new samples")
            new_samples = np.array(new_samples)
            # 对该时刻到达数据进行分裂形成粒球
            gb_list_temp = [new_samples]  # 粒球集合,初始只有一个粒球[ [[data1],[data2],...], [[data1],[data2],...],... ]
            while 1:
                ball_number_old = len(gb_list_temp)
                gb_list_temp = division_2_2(gb_list_temp)  # 粒球划分
                ball_number_new = len(gb_list_temp)
                if ball_number_new == ball_number_old:
                    break
            radius = []  # 汇总所有粒球半径
            for gb_data in gb_list_temp:
                if len(gb_data) >= 2:  # 粒球中样本多于2个时，认定为合法粒球，并收集其粒球半径
                    radius.append(get_radius(gb_data[:, :-1]))
            radius_median = np.median(radius)
            radius_mean = np.mean(radius)
            radius = min(radius_median, radius_mean)
            while 1:
                ball_number_old = len(gb_list_temp)
                gb_list_temp = minimum_ball(gb_list_temp, radius)  # 缩小粒球
                ball_number_new = len(gb_list_temp)
                if ball_number_new == ball_number_old:
                    break
            for obj in gb_list_temp:
                if len(obj) == 1:
                    continue
                gb = GranularBall(obj)
                if not self.micro_balls:
                    print("self.micro_balls is None")
                nearest_micro_ball_index, nearest_micro_ball, smallest_distance = \
                    self.get_nearest_micro_ball(gb.center, self.micro_balls)
                centers = [mb.center for mb in self.micro_balls]
                if nearest_micro_ball is None:
                    print("nearest_micro_ball is none")
                MIR = min((  # distance between the 1st closest center and the 2nd
                    # 离nearest_micro_ball最近的球 的距离
                    np.linalg.norm(center - nearest_micro_ball.center) for center in centers
                    if center is not nearest_micro_ball.center
                ))
                # if smallest_distance >= radius:
                if smallest_distance + gb.radius >= MIR and smallest_distance + gb.radius > nearest_micro_ball.radius and smallest_distance + nearest_micro_ball.radius > gb.radius:
                    mb = MicroBall(gb.data, label=None)
                    mb.init_weight(t, len(gb.data))
                    self.micro_balls.append(mb)
                else:
                    insert = nearest_micro_ball.insert_ball(gb, t)
                    if not insert:
                        continue
                    else:
                        del self.micro_balls[nearest_micro_ball_index]
                        self.micro_balls.extend(insert)
            temp = []
            for mb in self.micro_balls:
                # 更新权重
                mb.update_weight(t, self.lam)
                if mb.weight > self.threshold:
                    temp.append(mb)
            self.micro_balls = temp
            if int((t - self.begin) % 1) == 0:
                clusters, gb_list, gb_dict = self.connect_ball_DPC(t)
                if plot_evaluate_flag:
                    gb_plot(gb_dict, [], t)
            else:
                print(f"{(t - self.begin) % 0.1} not connect")
            t += self.timeSpan
            print("now have", len(self.micro_balls), "micro-balls")

    def connect_ball_DPC(self, t, init_data=None):
        """
        （zt）如果init_data不为空，则为初始化阶段，否则为中间阶段。
             以粒球中心为输入数据，连接粒球，生成聚类。
        :param init_data: 初始化数据。
        :return:clusters 初始化数据聚类结果{簇标签：[data1,data2,...]}
        """
        radius = []  # 汇总所有粒球半径
        gb_list_temp = []
        if init_data is not None:
            gb_list_temp = [init_data]  # 粒球集合,初始只有一个粒球[ [[data1],[data2],...], [[data1],[data2],...],... ]
            while 1:
                ball_number_old = len(gb_list_temp)
                gb_list_temp = division_2_2(gb_list_temp)  # 粒球划分
                ball_number_new = len(gb_list_temp)
                if ball_number_new == ball_number_old:
                    break
            for gb_data in gb_list_temp:
                if len(gb_data) >= 2:  # 粒球中样本多于2个时，认定为合法粒球，并收集其粒球半径
                    radius.append(get_radius(gb_data[:, :-1]))
        else:
            for mb in self.micro_balls:
                if len(mb.data) >= 2:
                    radius.append(mb.radius)
                gb_list_temp.append(mb.data)
        radius_median = np.median(radius)
        radius_mean = np.mean(radius)
        radius = max(radius_median, radius_mean)
        while 1:
            ball_number_old = len(gb_list_temp)
            gb_list_temp = normalized_ball(gb_list_temp, radius)  # 归一化
            ball_number_new = len(gb_list_temp)
            if ball_number_new == ball_number_old:
                break
        gb_center_list = []
        # noise
        gb_list_temp_no_noise = []
        for gb in gb_list_temp:
            # noise
            if len(gb) > 1:
                gb_center_list.append(gb[:, :-1].mean(0))
                gb_list_temp_no_noise.append(gb)
        gb_center = np.array(gb_center_list)
        # noise
        gb_list_temp = gb_list_temp_no_noise
        clusters_label, n = get_cluster_DPC(gb_center)
        print("==================t=" + str(t) + "=================")
        print("  c_num: " + str(n))
        clusters = {}
        gb_dict = {}
        for i in range(0, len(gb_list_temp)):
            gb_dict[i] = GB(gb_list_temp[i], clusters_label[i])
            if clusters_label[i] in clusters.keys():
                clusters[clusters_label[i]] = np.append(clusters[clusters_label[i]], gb_list_temp[i], axis=0)
            else:
                clusters[clusters_label[i]] = gb_list_temp[i]
        return clusters, gb_list_temp, gb_dict


def start(data, dataset_name, plot_evaluate_flag):
    M = MBStream(data, dataset_name, plot_evaluate_flag)
    M.fit_predict(plot_evaluate_flag)
