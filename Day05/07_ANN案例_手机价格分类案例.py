"""
案例:
    ANN(人工神经网络)案例:  手机价格预测

背景:
    基于手机20列第特征 --> 预测手机的价格区间(4个区间), 可以用机器学习做, 也可以用 深度学习做

ANN的案例实现步骤:
    1. 构建数据集
    2. 搭建神经网络
    3. 模型训练
    4. 模型测试
"""

# 导包
import torch                                        # Pytorch框架, 封装了张量的各种操作
from torch.utils.data import TensorDataset          # 数据集对象 数据 --> 张量 --> 数据集对象 --> 数据加载器
from torch.utils.data import DataLoader             # 数据加载器
import torch.nn as nn                               # 封装了各种神经网络
import torch.optim as optim                         # 优化器
from sklearn.model_selection import train_test_split    #训练集和测试集的划分
import matplotlib.pyplot as plt                     # 绘图
import numpy as np                                  # 数组
import pandas as pd                                 # 数据分析
import time                                         # 时间模块

# todo 1. 定义函数, 构建数据集
def creat_dataset():
    # 1.加载csv文件数据集
    data = pd.read_csv('./data/手机价格预测.csv')

    # 2.获取x特征列 和 y标签列
    x = data.iloc[:,:-1]
    y = data.iloc[:,-1]

    # 3.把特征列转出浮点数
    x = x.astype(np.float32)
    y = y.astype(np.int64)

    # 4.切分训练集和测试集
    # 参数1: x特征 参数2: y标签 参数3: 测试集比例 参数4: 随机种子 参数5: 样本的分布是否相同(即:参考y的类别进行抽取数据)
    x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2,random_state=3,stratify=y)

    # 5.把数据集转换成张量 思路: 数据 --> 张量 --> 数据集对象 --> 数据加载器
    train_dataset = TensorDataset(torch.tensor(x_train.values), torch.tensor(y_train.values))
    test_dataset = TensorDataset(torch.tensor(x_test.values), torch.tensor(y_test.values))

    # 6.返回                               20(充当输入特征数)   4(充当输出标签数)
    return train_dataset, test_dataset, x_train.shape[1],len(np.unique(y))

# todo 2. 搭建神经网络

# todo 3. 模型训练

# todo 4. 模型测试

# 测试
if __name__ == '__main__':
    train_dataset, test_dataset, input_dim, output_dim = creat_dataset()
    print(f'训练集的数据集对象: {train_dataset}')
    print(f'测试集的数据集对象: {test_dataset}')
    print(f'输入特征维度: {input_dim}')
    print(f'输出标签维度: {output_dim}')