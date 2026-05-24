"""
案例:
    代码演示 随机失活

正则化的作用:
    缓解模型的过拟合情况

正则化的方式:
    L1正则化:  权重可以变为0, 相当于: 降维
    L2正则化:  权重可以无限接近0
    DropOut:  随机失活, 每批次样本训练时, 随机让一部分神经元死亡,防止一些特征对结果的影响较大
    BN(批量归一化):  ...
"""

# 导包
import torch
import torch.nn as nn

# 定义函数, 演示随机失活
def dm01():
    # 1.创建输入特征
    x = torch.randint(0,10,(1,4),dtype=torch.float32)
    print(x)

    # 2.创建网络结构
    Linear1 = nn.Linear(4,5)

    # 3.加权求和
    l1 = Linear1(x)
    print(l1)

    # 4.激活函数
    output = torch.relu(l1)
    print(output)

    # 5.对激活函数进行DropOut随机失活
    dropout = nn.Dropout(p=0.5)

    # 6.具体的失活动作
    d1 = dropout(output)
    print(d1)

# 测试
if __name__ == '__main__':
    dm01()