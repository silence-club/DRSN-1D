import os
from scipy.io import loadmat
import numpy as np
import random
from sklearn.preprocessing import scale, StandardScaler, MinMaxScaler
from collections import Counter



# import matplotlib.pyplot as plt

def iteror_raw_data(data_path, data_mark):
    """
       打标签，并返回数据的生成器：标签，样本数据。

       :param data_path：.mat文件所在路径
       :param data_mark："FE" 或 "DE"
       :return iteror：（标签，样本数据）
    """

    # 标签数字编码
    labels = {"98": 0, "106": 1, "119": 2, "131": 3, "108": 4,
              "118": 5, "110": 6,"270": 7, "271": 8, "272": 9, "273": 10}

    # 列出所有文件
    filenames = os.listdir(data_path)

    # 逐个对mat文件进行打标签和数据提取
    for single_txt in filenames:

        single_mat_path = os.path.join(data_path, single_txt)
        # 打标签
        for key, _ in labels.items():
            if key in single_txt:
                label = labels[key]

        # 数据提取
        data = np.loadtxt(single_mat_path)
        '''for key, _ in file.items():
            if data_mark in key:
                data = file[key].ravel()  # series'''
        yield label, data


def data_augment(fs, win_tlen, overlap_rate, data_iteror):
    """
        :param win_tlen: 滑动窗口的时间长度
        :param overlap_rate: 重叠部分比例, [0-100]，百分数；
                             overlap_rate*win_tlen*fs//100 是论文中的重叠量。
        :param fs: 原始数据的采样频率
        :param data_iteror: 原始数据的生成器格式
        :return (X, y): X, 切分好的数据， y数据标签
                        X[0].shape == (win_len,)
                        X.shape == (N, win_len)
    """
    overlap_rate = int(overlap_rate)
    # 窗口的长度，单位采样点数
    win_len = int(fs * win_tlen)
    # 重合部分的时间长度，单位采样点数
    overlap_len = int(win_len * overlap_rate / 100)
    # 步长，单位采样点数
    step_len = int(win_len - overlap_len)

    # 滑窗采样增强数据
    X = []
    y = []
    for iraw_data in data_iteror:
        single_raw_data = iraw_data[1].ravel()
        lab = iraw_data[0]
        len_data = single_raw_data.shape[0]

        for start_ind, end_ind in zip(range(0, len_data - win_len, step_len),
                                      range(win_len, len_data, step_len)):
            X.append(single_raw_data[start_ind:end_ind].ravel())
            y.append(lab)

    X = np.array(X)
    y = np.array(y)

    return X, y


def under_sample_for_c0(X, y, low_c0, high_c0, random_seed):  # -> 没有使用
    """ 使用非0类别数据的数目，来对0类别数据进行降采样。
        :param X: 增强后的振动序列
        :param y: 类别标签0-9
        :param low_c0: 第一个类别0样本的索引下标
        :param high_c0: 最后一个类别0样本的索引下标
        :param random_seed: 随机种子
        :return X,y
    """

    np.random.seed(random_seed)
    to_drop_ind = random.sample(range(low_c0, high_c0), (high_c0 - low_c0 + 1) - len(y[y == 3]))
    # 按照行删除
    X = np.delete(X, to_drop_ind, 0)
    y = np.delete(y, to_drop_ind, 0)
    return X, y


def over_sample(X, y, len_data, random_seed=None):  # -> 没有使用
    """
    对样本较少的类别进行过采样，增加样本数目，实现样本平衡
    """
    oversampler = sv.MulticlassOversampling(sv.distance_SMOTE(random_state=random_seed))
    X_samp, y_samp = oversampler.sample(X, y)
    return X_samp, y_samp


def preprocess(path, data_mark, fs, win_tlen,
               overlap_rate, random_seed, **kargs):
    win_len = int(fs * win_tlen)
    data_iteror = iteror_raw_data(path, data_mark)
    X, y = data_augment(fs, win_tlen, overlap_rate, data_iteror, **kargs)
    # print(len(y[y==0]))

    print("-> 数据位置:{}".format(path))
    print("-> 原始数据采样频率:{0}Hz,\n-> 数据增强后共有：{1}条,"
          .format(fs, X.shape[0]))
    print("-> 单个数据长度：{0}采样点,\n-> 重叠量:{1}个采样点,"
          .format(X.shape[1], int(overlap_rate * win_tlen * fs // 100)))
    print("-> 类别数据数目:", sorted(Counter(y).items()))
    return X, y


