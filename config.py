import os
import re
from osgeo import gdal

"""
        1.以下代码每次调用config均会执行更新，以便读取data相关参数
        2.可修改newdata_path使用自己数据集，其余数据相关参数会自动更新
        3.所有data和label尺寸大小均需完全一致
"""

config = {
    "newdata_path": "./data/",  # 用户可修改存放数据的根目录，使用自己数据
    "data_path": ["origin_data/aspect_10.tif",
                  "origin_data/duanceng_5.tif",
                  "origin_data/elevation_5.tif",
                  "origin_data/gcyz_3.tif",
                  "origin_data/gougumd_5.tif",
                  "origin_data/qifudu_5.tif",
                  "origin_data/river_5.tif",
                  "origin_data/road_5.tif",
                  "origin_data/slope_5.tif",
                  "origin_data/slope_5.tif",
                  ],
    "label_path": "origin_data/label1.tif",
    # 标签TIF文件  需包括 0(训练集滑坡) 1(测试集滑坡) 2(训练集非滑坡) 3(测试集非滑坡)   0+1=2+3 且 (0+2):(1+3)=7:3 or 8:2
    "feature": 9,
    "width": 3368,
    "height": 2626,  # height * width
    "batch_size": 16392,  # 训练batch_size大小 
    "epochs": 250,  # 训练epoch数量
    "Cutting_window": 32,  # 切片大小为： Cutting window * height
    "device": "cuda",  # "cuda"  "cpu"
    "lr": 3e-3,
    "max_value": 24
}

data_path = []
for tif_data in os.listdir(config["newdata_path"]):
    if tif_data.endswith('tif'):
        if re.match('label', tif_data):
            config["label_path"] = config["newdata_path"] + tif_data  # 更新标签路径
            continue
        temp = config["newdata_path"] + tif_data
        data_path.append(temp)

config["data_path"] = data_path
config["feature"] = len(data_path)
tif = gdal.Open(config["data_path"][0])
config["width"], config["height"] = tif.RasterXSize, tif.RasterYSize
