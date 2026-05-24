"""
案例:
    代码演示批量归一化, 它属于正则化的一种, 也是用于 缓解模型的 过拟合情况的

批量归一化:
    思路:
        先对数据标准化,再对数据重构      数据标准化会丢失一些信息,通过数据重构找补回一些信息,缩放(入)+平移(β)
    应用场景:
        批量归一化在计算机视觉领域用的较多

    BatchNorm1d:    主要应用于全连接层或处理一维数据的网络, 例如文本处理. 它接收形状为(N, num_features) 的张量作为输入
    BatchNorm2d:    主要应用于卷积神经网络, 处理二维图像数据或特征图. 它接收形状为(N, C, H, W)的张量作为输入
    BatchNorm3d:    主要应用于三维卷积神经网络(3D CNN), 处理三维数据, 例如视频 或 医学图像.它接收形状为(N, C, D, H, W)的张量
"""

# 导包
import torch
import torch.nn as nn

# 1.定义函数, 处理二维数据
def dm01():
        # 1.创建图像样本数据
        input_2d = torch.randn(size=(1,2,3,4))
        print(f'input_2d: {input_2d}')

        # 2.创建批量归一化层(BN层)
        # 参1: 输入特征维度 = 图片通道数
        # 参2: 批量归一化的参数 eps: 避免除零错误, 默认值为1e-5
        # 参3: 批量归一化的参数 momentum: 动量, 默认值为0.1
        # 参4: 批量归一化的参数 affine: 是否可学习, 默认值为True (入,β) 缩放和平移
        bn2d = nn.BatchNorm2d(num_features=2,eps=1e-5,momentum=0.1,affine=True)

        # 3.对数据进行 批量归一化处理
        output_2d = bn2d(input_2d)
        print(f'output_2d: {output_2d}')

# 2.定义函数, 处理一维数据
def dm02():
    # 1.创建一维样本数据
    # 2行2列
    input_1d = torch.randn(size=(2,2))
    print(f'input_1d: {input_1d}')

    # 2.创建线性层
    linear1 = nn.Linear(2,4)

    # 3.对数据进行线性变换
    l1 = linear1(input_1d)

    # 4.创建批量归一化层(BN层)
    bn1d = nn.BatchNorm1d(num_features=4)

    # 5.对数据进行 批量归一化处理
    output_1d = bn1d(l1)
    print(f'output: {output_1d}')

# 测试
if __name__ == '__main__':
    # dm01()
    dm02()