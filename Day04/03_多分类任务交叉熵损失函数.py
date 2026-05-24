"""
案例:
    演示 多分类任务的交叉熵损失函数

函数介绍:
    概述:
        损失函数也叫成本函数,代价函数,目标函数,误差函数
    分类:
        分类问题:
            多分类交叉熵损失: CrossEntropyLoss
            二分类交叉熵损失:BCELoss

        回归问题:
            MAE: Mean Absolute Error ,平均绝对误差
            MSE: Mean Square Error , 均方误差
            Smooth L1: 结合上述两个的特点做的升级,优化

多分类交叉熵损失:CrossEntropyLoss
    设计思路:
        Loss = - Σylog(S(f(x)))
    简单记忆:
        x:          样本
        f(x):       加权求和
        S(f(x)):    处理后的概率
        y:          样本x属于某一个类别的 真实概率
    通俗解释:
        损失函数结果 = 正确类别概率的对数的最小化
    细节:
todo  CrossEntropyLoss = Softmax() + 损失计算  如果后续用这个损失函数,则:  输出层就不用额外调用Softmax()激活函数了
"""

# 导包
import torch
import torch.nn as nn

# 定义函数
def dm01():
    # 1.手动创建样本的真实值 - > y
    y_true = torch.tensor([[0,1,0],[1,0,0]],dtype=torch.float)

    # 2.手动创建样本的预测分数 - > f(x) 神经网络的原始输出分数(logits)，还没经过Softmax()激活函数归一化转换成概率
    logits = torch.tensor([[2,3,1],[8,2,7]],dtype=torch.float,requires_grad=True)

    # 3.创建损失函数对象 多分类交叉熵损失: CrossEntropyLoss = Softmax() + 损失计算
    criterion = nn.CrossEntropyLoss()   # 平均损失

    # 输入 logits (原始分数)
    #   第1步: Softmax 归一化 → 转换成概率分布
    #   第2步: 计算交叉熵 → 衡量预测概率与真实标签的差异
    #   输出 loss (标量值，越小越好)

    # 4.计算损失值
    loss = criterion(logits,y_true)
    print(f'loss:{loss}')

# 测试
if __name__ == '__main__':
    dm01()