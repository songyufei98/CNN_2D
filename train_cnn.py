import math
import os
import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch.nn.functional as F
from torch.utils.data import TensorDataset
import torch.nn as nn
from torch import optim
import numpy as np
from model.CNN2D import CNN_2D
from utils import drawAUC_TwoClass, plot_and_save, calculate_f1_score, calculate_auc, calculate_mse_rmse
from torch.optim.lr_scheduler import LambdaLR
from torch.autograd import Variable
from tensorboardX import SummaryWriter

import config

config = config.config


# train函数
def train(alldata_train, alltarget_train, alldata_val, alltarget_val):
    max_score = 0
    train_dataset = TensorDataset(torch.from_numpy(alldata_train).float(),torch.from_numpy(alltarget_train).float())
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=config["batch_size"], shuffle=True)
    val_dataset = TensorDataset(torch.from_numpy(alldata_val).float(), torch.from_numpy(alltarget_val).float())
    val_loader = torch.utils.data.DataLoader(dataset=val_dataset, batch_size=config["batch_size"], shuffle=True)

    model = CNN_2D().to(config["device"])

    model_name = model.__class__.__name__
    if "raw" in config["newdata_path"]:
        Data_Type = "Raw"
    else:
        Data_Type = "FR"
    result_folder = f"{model_name}_" + Data_Type

    writer = SummaryWriter(log_dir=os.path.join('Result', result_folder, 'log_dir'))

    # 损失函数与优化器
    criterion = nn.CrossEntropyLoss().to(config["device"])
    # 初始化优化器
    optimizer = optim.Adam(model.parameters(), lr=config['lr'])

    train_acc_list = []
    train_loss_list = []
    train_f1_list = []
    val_acc_list = []
    val_loss_list = []
    val_f1_list = []

    for epoch in range(config["epochs"]):
        train_acc = 0.0
        train_loss = 0.0
        train_f1 = 0.0
        train_mse = 0.0
        train_rmse = 0.0
        val_acc = 0.0
        val_loss = 0.0
        val_f1 = 0.0
        val_mse = 0.0
        val_rmse = 0.0
        train_outputs_list = []
        train_labels_list = []
        val_outputs_list = []
        val_labels_list = []

        model.train()
        for images, target in train_loader:
            # 反向传播
            images, target = Variable(images).to(config["device"]), Variable(target).to(config["device"])
            optimizer.zero_grad()
            outputs = model(images)
            _, preds = torch.max(outputs.data, 1)
            loss = criterion(outputs, target.squeeze().long())
            loss.backward()
            optimizer.step()
            train_outputs_list.extend(outputs.detach().cpu().numpy())
            train_labels_list.extend(target.cpu().numpy())
            train_array = np.array(train_outputs_list)
            train_mse, train_rmse = calculate_mse_rmse(outputs, target)
            train_acc += (preds[..., None] == target).squeeze().sum().cpu().numpy()
            train_loss += loss.item()
            train_f1 = calculate_f1_score(target.cpu().numpy(), preds.cpu().numpy())
            train_auc = calculate_auc(train_labels_list, train_array[:, 1])  # auc为Roc曲线下的面积

        writer.add_scalars('LOSS/', {'Train_Loss': train_loss / len(train_dataset)}, epoch)
        writer.add_scalars('ACC/', {'Train_Acc': float(train_acc) / len(train_dataset)}, epoch)

        model.eval()
        with torch.no_grad():
            for images, target in val_loader:
                images, target = Variable(images).to(config["device"]), Variable(target).to(config["device"])
                outputs = model(images)
                loss = criterion(outputs, target.squeeze().long())
                val_loss += loss.item()
                val_outputs_list.extend(outputs.detach().cpu().numpy())
                val_labels_list.extend(target.cpu().numpy())
                score_array = np.array(val_outputs_list)
                val_mse, val_rmse = calculate_mse_rmse(outputs, target)
                _, preds = torch.max(outputs.data, 1)
                val_acc += (preds[..., None] == target).squeeze().sum().cpu().numpy()
                val_f1 = calculate_f1_score(target.cpu().numpy(), preds.cpu().numpy())
                val_auc = calculate_auc(val_labels_list, score_array[:, 1])

            # 计算加权综合分数
            weight_score = 0.2 * (val_acc / len(val_dataset)) + 0.6 * (val_auc / 100) + 0.2 * val_f1

            print('[%03d/%03d]  Train Acc: %3.5f AUC: %3.5f Loss: %3.6f F1: %3.5f MSE: %3.5f RMSE: %3.5f'
                  '| Val Acc: %3.5f AUC: %3.5f Loss: %3.6f F1: %3.5f MSE: %3.5f RMSE: %3.5f'
                  '| Weight Score: %3.5f' % \
                  (epoch + 1, config["epochs"], train_acc / len(train_dataset), train_auc,
                   train_loss / len(train_dataset), train_f1, train_mse, train_rmse,
                   val_acc / len(val_dataset), val_auc, val_loss / len(val_dataset), val_f1, val_mse, val_rmse, weight_score))

            # save result for each epoch
            file_path = os.path.join('Result', result_folder, f"{result_folder}.txt")
            with open(file_path, 'a') as f:
                f.write('[%03d/%03d]  Train Acc: %3.5f AUC: %3.5f Loss: %3.6f F1: %3.5f MSE: %3.5f RMSE: %3.5f'
                        '| Val Acc: %3.5f AUC: %3.5f Loss: %3.6f F1: %3.5f MSE: %3.5f RMSE: %3.5f'
                        '| Weight Score: %3.5f\n' % \
                        (epoch + 1, config["epochs"], train_acc / len(train_dataset), train_auc,
                         train_loss / len(train_dataset), train_f1, train_mse, train_rmse,
                         val_acc / len(val_dataset), val_auc, val_loss / len(val_dataset), val_f1, val_mse, val_rmse, weight_score))
            train_acc_list.append(train_acc / len(train_dataset))
            train_loss_list.append(train_loss / len(train_dataset))
            train_f1_list.append(train_f1)
            val_acc_list.append(val_acc / len(val_dataset))
            val_loss_list.append(val_loss / len(val_dataset))
            val_f1_list.append(val_f1)

            if weight_score > max_score:
                max_score = weight_score
                drawAUC_TwoClass(train_labels_list, train_array[:, 1],
                                 os.path.join('Result', result_folder, 'train_AUC.png'))
                drawAUC_TwoClass(val_labels_list, score_array[:, 1],
                                 os.path.join('Result', result_folder, 'val_AUC.png'))
                train_auc = calculate_auc(train_labels_list, train_array[:, 1])  # auc为Roc曲线下的面积
                val_auc = calculate_auc(val_labels_list, score_array[:, 1])
                best_result = os.path.join('Result', result_folder, "best_result.txt")
                with open(best_result, 'w') as f:
                    f.write('Train Acc: %3.5f AUC: %3.5f Loss: %3.6f F1: %3.5f MSE: %3.5f RMSE: %3.5f'
                            '| Val Acc: %3.5f AUC: %3.5f Loss: %3.6f F1: %3.5f MSE: %3.5f RMSE: %3.5f'
                            '| Weight Score: %3.5f' % \
                            (train_acc / len(train_dataset), train_auc, train_loss / len(train_dataset), train_f1,
                                train_mse, train_rmse,
                                val_acc / len(val_dataset), val_auc, val_loss / len(val_dataset), val_f1, val_mse,
                                val_rmse, weight_score))
                torch.save(model.state_dict(), os.path.join('Result', result_folder, 'best.pth'))
            # 记录Loss, accuracy
            writer.add_scalars('LOSS/valid', {'valid_loss': val_loss / len(val_dataset)}, epoch)
            writer.add_scalars('ACC/valid', {'valid_acc': val_acc / len(val_dataset)}, epoch)
    plot_and_save(os.path.join('Result', result_folder), train_acc_list, train_loss_list,
                  train_f1_list, val_acc_list, val_loss_list, val_f1_list)
    torch.save(model.state_dict(), os.path.join('Result', result_folder, 'latest.pth'))
