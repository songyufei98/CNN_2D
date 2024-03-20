import read_data
import config
import generate_LSM
import train_cnn
#实例化config
config = config.config
from read_data import get_feature_data, creat_dataset

is_train = 0
is_saveLSM = 1

if __name__ == '__main__':
    if is_train:
        print('***************************************读取训练集 测试集***************************************')
        tensor_data, max_value = get_feature_data()
        config["max_value"] = max_value  # 离散化因子最大值作为2D数据的边长
        creat = creat_dataset(tensor_data, max_value)
        all_data_train, all_target_train = read_data.train_data(creat)
        all_data_val, all_target_val = read_data.test_data(creat)
        print('*******************************************开始训练*******************************************')
        train_cnn.train(all_data_train, all_target_train, all_data_val, all_target_val)
    if is_saveLSM:
        generate_LSM.save_LSM()