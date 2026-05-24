"""
案例:
    演示池化层相关操作

池化层解释(Pooling):
    目的:
        降维
    思路:
        最大池化
        平均池化
    特点:
        池化不会改变数据的 通道数
"""

# 导包
import torch
import torch.nn as nn

# 1. 定义函数, 演示单通道池化
def dm01():
    # 1. 创建1个 1通道 3*3 的二维张量
    inputs = torch.tensor([     # 1 通道C
        [                       # 3 高度H
            [1, 2, 3],          # 3 宽度W
            [4, 5, 6],
            [7, 8, 9]
        ]
    ])
    # print(f'inputs: {inputs}, inputs.shape: {inputs.shape}')  # [1, 3, 3]

    # 2. 创建最大池化层
    # 参1: 池化核大小 参2: 步长 参3: 填充
    pool = nn.MaxPool2d(2,1,0)
    outputs = pool(inputs)
    print(f'outputs: {outputs}, outputs.shape: {outputs.shape}')      # [1, 2, 2]

    # 3. 创建平均池化层
    pool = nn.AvgPool2d(2,1,0)
    outputs = pool(inputs)
    # print(f'outputs: {outputs}, outputs.shape: {outputs.shape}')      # [1, 2, 2]

# 2. 定义函数, 演示多通道池化
def dm02():
    # 1. 创建1个 1通道 3*3 的二维张量
    inputs = torch.tensor([     # 3 通道C
        [                       # 3 高度H
            [1, 2, 3],          # 3 宽度W
            [4, 5, 6],
            [7, 8, 9]
        ],
        [                       # 3 宽度W
            [10, 20, 30],       # 3 宽度W
            [40, 50, 60],
            [70, 80, 90]
        ],
        [                       # 3 宽度W
            [11, 22, 33],          # 3 宽度W
            [44, 55, 66],
            [77, 88, 99]
        ]
    ])
    # print(f'inputs: {inputs}, inputs.shape: {inputs.shape}')  # [3, 3, 3]

    # 2. 创建最大池化层
    # 参1: 池化核大小 参2: 步长 参3: 填充
    pool = nn.MaxPool2d(2,1,0)
    outputs = pool(inputs)
    print(f'outputs: {outputs}, outputs.shape: {outputs.shape}')      # [3, 2, 2]

    # 3. 创建平均池化层
    pool = nn.AvgPool2d(2,1,0)
    outputs = pool(inputs)
    # print(f'outputs: {outputs}, outputs.shape: {outputs.shape}')      # [3, 2, 2]

# 测试
if __name__ == '__main__':
    # dm01()
    dm02()