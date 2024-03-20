import os
import warnings
import numpy as np
import pandas as pd
import random
from osgeo import gdal
from PIL import Image
import config

config = config.config
import pickle


def map_values_to_int(array):
    w, h = array.shape[0], array.shape[1]
    # 将二维数组转换为一维数组
    array = array.flatten()
    # 使用numpy的unique函数获取唯一值和它们的索引
    _, indices = np.unique(array, return_inverse=True)
    # 将一维数组重新转换为二维数组
    mapped_array = indices.reshape(w, h)
    return mapped_array


def resample_tif(img_re):
    """
    :param img: original factors data
    :return: resampled factors data
    """
    warnings.filterwarnings("ignore")
    img_re = np.array(Image.fromarray(img_re).resize((config["height"], config["width"])))
    return img_re


def read_data_from_tif(tif_path):
    """
    读取影响因子数据并转换为nparray
    """
    tif = gdal.Open(tif_path)
    w, h = tif.RasterXSize, tif.RasterYSize
    img = np.array(tif.ReadAsArray(0, 0, w, h).astype(np.float32))
    if w != config["width"] and h != config["height"]:
        img = resample_tif(img)
    return img


def get_feature_data():
    """"
    读取特征并进行整数化
    """
    tif_paths = config["data_path"]
    max_values = []
    data = np.zeros((config["feature"], config["height"], config["width"])).astype(np.float32)
    for i, tif_path in enumerate(tif_paths):
        img = read_data_from_tif(tif_path)
        img[img == -np.finfo(np.float32).max] = 0
        # 将二维数据重新离散化
        data[i, :, :] = map_values_to_int(img)
        # 获取离散化后的因子最大值
        max_values.append(np.max(data[i, :, :]))
    max_value = int(max(max_values))
    return data, max_value


class creat_dataset():

    def __init__(self, tensor_data, size):
        self.data = tensor_data
        self.F = tensor_data.shape[0]
        self.h = tensor_data.shape[1]
        self.w = tensor_data.shape[2]
        self.size = size
        self.all_results = None

    def creat_new_tensor(self):
        # 扩大图像边缘
        new_tensor = np.zeros((self.F, self.h, self.w))
        # 将数据值赋值到图像中心
        new_tensor[:, 0:self.h, 0:self.w] = self.data
        return new_tensor

    def get_images_labels(self, data, labels, mode='train'):
        train_images, train_labels = [], []
        valid_images, valid_labels = [], []
        count_0, count_1, count_2, count_3 = 0, 0, 0, 0
        if self.all_results is not None:
            train_images, train_labels, valid_images, valid_labels = self.all_results
            if mode == "train":
                print('训练集： ' + str(len(train_images)), str(len(train_labels)))
                return train_images, train_labels
            else:
                print('测试集： ' + str(len(valid_images)), str(len(valid_labels)))
                return valid_images, valid_labels
        for i in range(config["height"]):
            for j in range(config["width"]):
                # 训练集
                if labels[i, j] == 0 or labels[i, j] == 2:
                    matrix = np.zeros((self.size, self.size)).astype(np.float32)
                    matrix[(data[:, i, j] - 1).astype(np.int32), np.arange(data[:, i, j].size)] = 1
                    train_images.append(matrix)
                    # 滑坡点
                    if labels[i, j] == 0:
                        count_0 += 1
                        train_labels.append(1)
                    # 非滑坡点
                    if labels[i, j] == 2:
                        count_2 += 1
                        train_labels.append(0)
                # 验证集
                if labels[i, j] == 1 or labels[i, j] == 3:
                    matrix = np.zeros((self.size, self.size)).astype(np.float32)
                    matrix[(data[:, i, j] - 1).astype(np.int32), np.arange(data[:, i, j].size)] = 1
                    valid_images.append(matrix)
                    # 滑坡点
                    if labels[i, j] == 1:
                        count_1 += 1
                        valid_labels.append(1)
                    # 非滑坡点
                    if labels[i, j] == 3:
                        count_3 += 1
                        valid_labels.append(0)
        print("label 为 0，1，2，3的像素点个数分别为{},{},{},{}".format(count_0, count_1, count_2, count_3))
        if self.all_results is None:
            self.all_results = train_images, train_labels, valid_images, valid_labels
        if mode == "train":
            print('训练集： ' + str(len(train_images)), str(len(train_labels)))
            return train_images, train_labels
        else:
            print('测试集： ' + str(len(valid_images)), str(len(valid_labels)))
            return valid_images, valid_labels


def get_train_data(config, creat):
    data = creat.creat_new_tensor()  
    labels = read_data_from_tif(config["label_path"])
    train_images, train_labels = creat.get_images_labels(data, labels, mode='train')
    return train_images, train_labels


def get_test_data(config, creat):
    data = creat.creat_new_tensor()
    labels = read_data_from_tif(config["label_path"])
    valid_images, valid_labels = creat.get_images_labels(data, labels, mode='valid')
    return valid_images, valid_labels


def shuffle_image_label_0(images, labels):
    """
    Randomly disrupt two list with the same shuffle
    """
    # 将两个列表（images和labels）随机打乱顺序，同时保持它们之间的对应关系
    # 这通常用于数据增强或者数据集准备阶段，以提高模型的泛化能力和训练效果。
    randnum = random.randint(0, len(images) - 1)
    random.seed(randnum)
    random.shuffle(images)
    random.seed(randnum)
    random.shuffle(labels)
    return images, labels


def train_data(creat):
    data_file = 'train_data.pkl'
    if os.path.exists(data_file):
        saved_config_dict = {}
        with open(data_file, 'rb') as f:
            data = pickle.load(f)
        saved_config = data.get('config', {})
        for item in saved_config:
            key = item[0]
            value = item[1]
            saved_config_dict[key] = value
        keys = ['label_path', 'feature', 'width', 'height', 'max_value']
        values = [config.get(key) for key in keys]
        saved_values = [saved_config_dict.get(key) for key in keys]
        if np.all(values == saved_values):
            train_images = data['train_images']
            train_labels = data['train_labels']
        else:
            train_images, train_labels = get_train_data(config, creat)
            config_list = [[k, v] for k, v in config.items()]
            data = {'train_images': train_images, 'train_labels': train_labels, 'config': config_list}
            with open(data_file, 'wb') as f:
                pickle.dump(data, f)
    else:
        train_images, train_labels = get_train_data(config, creat)
        config_list = [[k, v] for k, v in config.items()]
        data = {'train_images': train_images, 'train_labels': train_labels, 'config': config_list}
        with open(data_file, 'wb') as f:
            pickle.dump(data, f)
    train_images, train_labels = shuffle_image_label_0(train_images, train_labels)
    return np.array(train_images).reshape((-1, 1, config["max_value"], config["max_value"])), np.array(
            train_labels).reshape((-1, 1))


def test_data(creat):
    data_file = 'valid_data.pkl'
    if os.path.exists(data_file):
        saved_config_dict = {}
        with open(data_file, 'rb') as f:
            data = pickle.load(f)
        saved_config = data.get('config', {})
        for item in saved_config:
            key = item[0]
            value = item[1]
            saved_config_dict[key] = value
        keys = ['label_path', 'feature', 'width', 'height', 'max_value']
        values = [config.get(key) for key in keys]
        saved_values = [saved_config_dict.get(key) for key in keys]
        if np.all(values == saved_values):
            valid_images = data['valid_images']
            valid_labels = data['valid_labels']
        else:
            valid_images, valid_labels = get_test_data(config, creat)
            config_list = [[k, v] for k, v in config.items()]
            data = {'valid_images': valid_images, 'valid_labels': valid_labels, 'config': config_list}
            with open(data_file, 'wb') as f:
                pickle.dump(data, f)
    else:
        valid_images, valid_labels = get_test_data(config, creat)
        config_list = [[k, v] for k, v in config.items()]
        data = {'valid_images': valid_images, 'valid_labels': valid_labels, 'config': config_list}
        with open(data_file, 'wb') as f:
            pickle.dump(data, f)
    return np.array(valid_images).reshape((-1, 1, config["max_value"], config["max_value"])), np.array(
            valid_labels).reshape((-1, 1))


def save_to_tif(pred_result, save_path):
    """
    :保存LSM
    """
    img = pred_result.reshape((config["height"], config["width"]))
    im_geotrans, im_prof = [], []
    for tif_path in config["data_path"]:  # 取仿射矩阵、投影坐标
        tif = gdal.Open(tif_path)
        im_geotrans.append(tif.GetGeoTransform())
        im_prof.append(tif.GetProjection())

    if 'int8' in img.dtype.name:
        datatype = gdal.GDT_Byte
    elif 'int16' in img.dtype.name:
        datatype = gdal.GDT_UInt16
    else:
        datatype = gdal.GDT_Float32

    im_height, im_width = img.shape

    # 创建文件
    driver = gdal.GetDriverByName("GTiff")  # 数据类型必须有，因为要计算需要多大内存空间
    dataset = driver.Create(save_path, im_width, im_height, 1, datatype)
    dataset.GetRasterBand(1).WriteArray(img)  # 写入数组数据
    dataset.SetGeoTransform(im_geotrans[0])  # 写入仿射变换参数
    dataset.SetProjection(im_prof[0])  # 写入投影
    del dataset
    print('ok')
