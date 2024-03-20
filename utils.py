import math
import torch
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, f1_score
import torch.nn.functional as F
from torch import nn
import config

config = config.config

def calculate_mse_rmse(outputs, target):
    outputs_softmax = F.softmax(outputs, dim=1)
    target_one_hot = F.one_hot(target.long(), num_classes=2).squeeze()
    mse_loss = F.mse_loss(outputs_softmax, target_one_hot.float())
    rmse = torch.sqrt(mse_loss)
    return mse_loss.item(), rmse.item()



def random_flip(img, horizontal_prob=0.5, vertical_prob=0.5):
    # 水平翻转
    if np.random.rand() < horizontal_prob:
        img = img[:, :, ::-1]
    # 垂直翻转
    if np.random.rand() < vertical_prob:
        img = img[:, ::-1, :]

    return img


class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class SPPF(nn.Module):
    # Spatial Pyramid Pooling - Fast (SPPF)
    def __init__(self, c1, c2, k=5):  # equivalent to SPP(k=(5, 9, 13))
        super().__init__()
        c_ = c1 // 2  # hidden channels
        self.cv1 = nn.Sequential(
            nn.Conv2d(c1, c_, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(c_),
            nn.SiLU())
        self.cv2 = nn.Sequential(
            nn.Conv2d(c_ * 4, c2, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(c2),
            nn.SiLU())
        self.m = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)

    def forward(self, x):
        x = self.cv1(x)
        y1 = self.m(x)
        y2 = self.m(y1)
        return self.cv2(torch.cat((x, y1, y2, self.m(y2)), 1))


# 修正后的SPP-----> 能够使其处理不同大小的输入
class Modified_SPPLayer(torch.nn.Module):

    def __init__(self, num_levels, pool_type='max_pool'):
        # super(SPPLayer, self).__init__()
        self.num_levels = num_levels
        self.pool_type = pool_type

    def forward(self, x):
        # num:样本数量 c:通道数 h:高 w:宽
        # num: the number of samples
        # c: the number of channels
        # h: height
        # w: width
        num, c, h, w = x.size()
        #         print(x.size())
        for i in range(self.num_levels):
            level = i + 1
            kernel_size = (math.ceil(h / level), math.ceil(w / level))
            stride = (math.floor(h / level), math.floor(w / level))
            pooling = (
                math.floor((kernel_size[0] * level - h + 1) / 2), math.floor((kernel_size[1] * level - w + 1) / 2))

            # update input data with padding
            zero_pad = torch.nn.ZeroPad2d((pooling[1], pooling[1], pooling[0], pooling[0]))
            x_new = zero_pad(x)

            # update kernel and stride
            h_new = 2 * pooling[0] + h
            w_new = 2 * pooling[1] + w

            kernel_size = (math.ceil(h_new / level), math.ceil(w_new / level))
            stride = (math.floor(h_new / level), math.floor(w_new / level))

            # 选择池化方式
            if self.pool_type == 'max_pool':
                try:
                    tensor = F.max_pool2d(x_new, kernel_size=kernel_size, stride=stride).view(num, -1)
                except Exception as e:
                    print(str(e))
                    print(x.size())
                    print(level)
            else:
                tensor = F.avg_pool2d(x_new, kernel_size=kernel_size, stride=stride).view(num, -1)

            # 展开、拼接
            if i == 0:
                x_flatten = tensor.view(num, -1)
            else:
                x_flatten = torch.cat((x_flatten, tensor.view(num, -1)), 1)
        return x_flatten


# AUC绘制函数
def drawAUC_TwoClass(y_true, y_score, path):
    fpr, tpr, thresholds = roc_curve(y_true, y_score)
    roc_auc = auc(fpr, tpr)  # auc为Roc曲线下的面积
    roc_auc = roc_auc * 100
    # 开始画ROC曲线
    plt.figure(figsize=(5, 5), dpi=300)
    plt.plot(fpr, tpr, color='darkorange', linestyle='-', linewidth=2,
             label=('CNN (' + str(path).split('.')[0] + ' = %0.2f %%)' % roc_auc))
    plt.legend(loc='lower right')  # 设定图例的位置，右下角
    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.tick_params(direction='in', top=True, bottom=True, left=True, right=True)  # 坐标轴朝向
    plt.yticks(np.arange(0, 1.1, 0.1))
    plt.xticks(np.arange(0, 1.1, 0.1))

    plt.grid(linestyle='-.')
    plt.xlabel('False Positive Rate')  # 横坐标是fpr
    plt.ylabel('True Positive Rate')  # 纵坐标是tpr
    plt.legend(loc="lower right")

    # print("AUC:",roc_auc)
    plt.savefig(path, format='png')
    plt.close()


def calculate_f1_score(y_true, y_pred):
    """
    计算F1分数

    参数:
    y_true : 真实标签列表
    y_pred : 预测标签列表

    返回:
    f1 : F1分数
    """
    f1 = f1_score(y_true, y_pred, average='macro')
    return f1


def calculate_auc(y_true, y_score):
    fpr, tpr, thresholds = roc_curve(y_true, y_score)
    roc_auc = auc(fpr, tpr) * 100  # auc为Roc曲线下的面积
    return roc_auc


def plot_and_save(path, train_acc_list, train_loss_list, train_f1_list, val_acc_list, val_loss_list, val_f1_list):
    epochs = range(1, len(train_acc_list) + 1)
    # 绘制曲线并保存为 png 图片
    plt.clf()
    plt.figure(figsize=(7, 5), dpi=200)
    plt.subplots_adjust(left=0.3)
    plt.plot(epochs, train_acc_list, color='darkorange', linestyle='-', linewidth=2, label='Train Acc')
    plt.legend(loc='lower right')  # 设定图例的位置，右下角
    plt.xlabel('Epoch')
    plt.ylabel('Train Acc')
    plt.savefig(os.path.join(path, 'train_Acc.png'), format='png')

    plt.clf()
    plt.figure(figsize=(7, 5), dpi=200)
    plt.plot(epochs, train_loss_list, color='darkorange', linestyle='-', linewidth=2, label='Train Loss')
    plt.legend(loc='upper right')  # 设定图例的位置，右下角
    plt.xlabel('Epoch')
    plt.ylabel('Train Loss')
    plt.savefig(os.path.join(path, 'train_Loss.png'), format='png')

    plt.clf()
    plt.figure(figsize=(7, 5), dpi=200)
    plt.plot(epochs, train_f1_list, color='darkorange', linestyle='-', linewidth=2, label='Train F1')
    plt.legend(loc='upper right')  # 设定图例的位置，右下角
    plt.xlabel('Epoch')
    plt.ylabel('Train F1')
    plt.savefig(os.path.join(path, 'train_F1.png'), format='png')

    plt.clf()
    plt.figure(figsize=(7, 5), dpi=200)
    plt.subplots_adjust(left=0.3)
    plt.plot(epochs, val_acc_list, color='darkorange', linestyle='-', linewidth=2, label='Val Acc')
    plt.legend(loc='lower right')  # 设定图例的位置，右下角
    plt.xlabel('Epoch')
    plt.ylabel('Val Acc')
    plt.savefig(os.path.join(path, 'val_Acc.png'), format='png')

    plt.clf()
    plt.figure(figsize=(7, 5), dpi=200)
    plt.plot(epochs, val_loss_list, color='darkorange', linestyle='-', linewidth=2, label='Val Loss')
    plt.legend(loc='upper right')  # 设定图例的位置，右下角
    plt.xlabel('Epoch')
    plt.ylabel('Val Loss')
    plt.savefig(os.path.join(path, 'val_Loss.png'), format='png')

    plt.clf()
    plt.figure(figsize=(7, 5), dpi=200)
    plt.plot(epochs, val_f1_list, color='darkorange', linestyle='-', linewidth=2, label='Val F1')
    plt.legend(loc='upper right')  # 设定图例的位置，右下角
    plt.xlabel('Epoch')
    plt.ylabel('Val F1')
    plt.savefig(os.path.join(path, 'val_F1.png'), format='png')


def plot_save_lsm(path, probs):
    probs = probs.reshape((config["height"], config["width"]))
    # 数据可视化
    plt.figure(dpi=300)  
    plt.imshow(probs, cmap='hot', interpolation='nearest')
    plt.colorbar()  # 添加颜色条
    plt.savefig(os.path.join(path, 'CNN2D_LSM.png'), format='png', dpi=300)  
    plt.show()