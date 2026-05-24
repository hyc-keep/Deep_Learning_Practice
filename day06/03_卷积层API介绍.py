"""
案例:
    演示卷积层的API, 用于 提取图像的局部特征, 获取: 特征图(Feature Map)

卷积神经网络介绍:
    概述:
        全称叫: Convolutional neural network
    组成:
        卷积层(Convolutional):
            用于提取图像的 局部特征, 结合 卷积核(每个卷积核= 1个神经元) 实现, 处理后的结果叫: 特征图
        池化层(Pooling):
            用于 降维, 降采样
        全连接层(Full Connected, FC, Linear):
            用于 预测结果, 并输出结果的
    特征图计算方式:
        N = (W - F + 2P) / S + 1
        W: 输入图像的大小
        F: 卷积核的大小
        P: 填充的大小
        S: 步长
        N: 输出图像的大小(特征图大小)
"""

# 导包
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

# 定义函数, 用于完成图像的加载, 卷积, 特征图可视化操作
def dm01():
    # 1. 加载RGB真彩图
    img1 = plt.imread('./data/img.jpg')
    # 2. 打印读取到的图片信息
    # print(f'img1: {img1}, img1.shape: {img1.shape}')    #HWC: [640, 640, 3]

    # 3. 把图像的形状从 HWC --> CHW, 思路: img --> 张量 --> 转换维度
    img2 = torch.tensor(img1, dtype=torch.float32)
    img2 = img2.permute(2, 0, 1)
    # print(f'img2: {img2}, img2.shape: {img2.shape}')    # CHW: [3, 640, 640]

    # 4. 因为卷积层要求输入的图片的维度是4, 所以需要增加一个维度, 添加批次维度, 从CHW --> BCHW (1, C, H, W), 1张3通道640*640的图像
    img3 = img2.unsqueeze(dim=0)    # dim=0 表示在第0个维度添加批次维度 (1, C, H, W)
    # print(f'img3: {img3}, img3.shape: {img3.shape}')    # BCHW: [1, 3, 640, 640]

    # 5. 创建卷积层对象, 提取 图像的局部特征
    # 参1: 输入通道数 参2: 输出通道数(卷积核个数) 参3: 卷积核大小 参4: 步长 参5: 填充大小
    conv1 = nn.Conv2d(in_channels=3, out_channels=4, kernel_size=3, stride=2, padding=0)

    # 6. 具体的卷积操作
    output = conv1(img3)

    # 7. 打印卷积结果 输出结果: output.shape: [1, 4, 319, 319]
    # print(f'output: {output}, output.shape: {output.shape}')

    # 8. 查看提取到的 4个 特征图
    img4 = output[0]    #获得第一张图片的 4个特征图
    # print(f'img4: {img4}, img4.shape: {img4.shape}')    #CHW --> [4, 319, 319]

    # 9. 把上述的图 CHW --> HWC
    img5 = img4.permute(1, 2, 0)
    # print(f'img5: {img5}, img5.shape: {img5.shape}')    #HWC --> [319, 319, 4]

    # 10. 可视化第1个通道的特征图
    feature1 = img5[:, :, 0].detach().numpy()    # 获得第1个通道的特征图
    plt.imshow(feature1)
    plt.show()

# 测试
if __name__ == '__main__':
    dm01()