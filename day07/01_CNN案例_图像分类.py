"""
案例:
    演示CNN的综合案例, 图像分类

回顾: 深度学习项目的步骤
    1. 准备数据集
        用的是 计算机视觉模块 torchvision 自带的 CIFAR10 数据集, 包含6W张(32, 32, 3)的图片,5W张训练集,1W张测试集,10个分类,每个分类6K张图片
        需要单独安装 torchvision包, 即: pip install torchvision
    2. 搭建(卷积)神经网络
    3. 模型训练
    4. 模型测试

卷积层:
    提取图像的局部特征 --> 特征图(Feature Map), 计算方式: N = (W - F + 2P) // S + 1
    每个卷积核都是1个神经元

池化层:
    降维, 有最大池化 和 平均池化
    池化只在 HW 上做调整, 不改变通道数

案例的优化思路:
    1. 增加卷积核的输出通道数(卷积核的数量)
    2. 增加去全连接层的参数量
    3. 调整学习率
    4. 调整优化器(optimizer)
    5. 调整激活函数
    6. ...
"""

# 导包
import torch
import torch.nn as nn
from torchvision.datasets import CIFAR10
from torchvision.transforms import ToTensor
import torch.optim as optim
from torch.utils.data import DataLoader
import time
import matplotlib.pyplot as plt
from torchsummary import summary

# 每批次样本数
BATCH_SIZE = 8

# 1. 准备数据集
def create_dataset():
    # 1. 获取训练集
    # 参1: 数据集路径 参2: 是否是训练集 参3: 数据预处理 --> 张量数据 参4: 是否联网下载(若没有数据的话)
    train_dataset = CIFAR10(root='./data', train=True, transform=ToTensor(), download=True)
    # 2. 获取测试集
    test_dataset = CIFAR10(root='./data', train=False, transform=ToTensor(), download=True)
    # 3. 返回数据集
    return train_dataset, test_dataset

# 2. 搭建(卷积)神经网络
class ImageModel(nn.Module):
    # 1. 初始化父类成员, 搭建神经网络
    def __init__(self):
        # 1.1 初始化父类成员
        super().__init__()
        # 1.2 搭建神经网络
        # 第1个卷积层    参1: 输入 3通道 参2: 输出 6通道(卷积核个数)  参3: 卷积核大小3*3 参4: 步长1 参5: 填充0
        self.conv1 = nn.Conv2d(3,6,3,1,0)
        # 第1个池化层    参1: 池化核大小2*2 参2: 步长2 参3: 填充1
        self.pool1 = nn.MaxPool2d(2,2,0)

        # 第2个卷积层    参1: 输入 6通道 参2: 输出 16通道(卷积核个数) 参3: 卷积核大小3*3 参4: 步长1 参5: 填充0
        self.conv2 = nn.Conv2d(6,16,3,1,0)
        # 第2个池化层    参1: 池化核大小2*2 参2: 步长2 参3: 填充0
        self.pool2 = nn.MaxPool2d(2,2,0)

        # 第1个隐藏层(全连接层)  # 参1: 输入 576 参2: 输出 120
        self.linear1 = nn.Linear(576,120)

        # Dropout层1, 参1: 丢弃概率0.5
        self.dropout1 = nn.Dropout(0.5)

        # 第2个隐藏层(全连接层)  # 参1: 输入 120 参2: 输出 84
        self.linear2 = nn.Linear(120,84)

        # Dropout层2, 参1: 丢弃概率0.3(第二层小一点)
        self.dropout2 = nn.Dropout(0.3)

        # 第3个隐藏层(全连接层) --> 输出层 # 参1: 输入 84 参2: 输出 10
        self.output = nn.Linear(84,10)

    # 2. 前向传播
    def forward(self, x):
       # 第1层: 卷积层(加权求和) + 激励层(激活函数) + 池化层(降维)
        x = self.pool1(torch.relu(self.conv1(x)))

        # 第2层: 卷积层(加权求和) + 激励层(激活函数) + 池化层(降维)
        x = self.pool2(torch.relu(self.conv2(x)))

        # 细节: 全连接层只能处理二维数据,所以需要将数据拉平 --> view:改变形状(二维), (8, 16, 6, 6) --> (8, 576)
        # 参1: 样本数(行数) 参2: 特征数(列数)   -1:自适应(根据实际情况)
        x = x.view(x.size(0), -1)       # 8行576列
        # print(f'x.shape: {x.shape}')

        # dropout正则化, 防止过拟合(训练集准确率远高于测试集,产生了过拟合)
        # 第3层: 全连接层(加权求和) + 激励层(激活函数)
        x = torch.relu(self.linear1(x))
        x = self.dropout1(x)

        # 第4层: 全连接层(加权求和)  + 激励层(激活函数)
        x = torch.relu(self.linear2(x))
        x = self.dropout2(x)

        # 第5层: 全连接层(加权求和) --> 输出层
        x = self.output(x)      # 因为后续用交叉熵损失函数CrossEntropyLoss = softmax() + 损失计算,所以这里不需要激活函数

        return  x

# 3. 模型训练
def train(train_dataset):
    # 1. 创建数据加载器
    dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    # 2. 创建模型对象
    model = ImageModel()
    # 3. 创建损失函数对象
    criterion = nn.CrossEntropyLoss()
    # 4. 创建优化器对象
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    # 5. 循环遍历epoch, 开始 每轮的 训练动作
    # 5.1 定义变量, 记录训练的 总轮数
    epochs = 10
    # 5.2 遍历, 完成每轮的 所有批次的 具体训练动作
    for epoch in range(epochs):
        # 5.2.1 定义变量, 记录: 总损失, 总样本数, 预测正确的样本数, 训练的开始时间
        total_loss, total_samples, correct_samples, start_time = 0.0, 0, 0, time.time()
        # 5.2.2 遍历数据加载器, 获取当前批数据
        for x, y in dataloader:
            # 5.2.3 切换训练模式
            model.train()
            # 5.2.4 模型预测
            y_pred = model(x)
            # 5.2.5 计算损失
            loss = criterion(y_pred, y)
            # 5.2.6 梯度清零 + 反向传播 + 参数更新
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # 5.2.7 统计预测正确的样本数
            # torch.argmax(y_pred, dim=1): 获取预测结果中最大概率的索引, 得到输出类别,然后与真实标签进行比较,得到预测正确的样本数
            correct_samples += (torch.argmax(y_pred, dim=1) == y).sum()

            # 5.2.8 统计总损失
            # loss.item(): 获取每批的平均损失, 然后乘以样本数,得到本批次的总损失
            total_loss += loss.item() * x.shape[0]

            # 5.2.9 统计总样本数
            # x.shape[0]: 获取每批样本数
            total_samples += x.shape[0]

        # 5.2.10 走到这里, 说明本批次训练完成, 开始打印本批次的训练结果
        print(f'第 {epoch+1}轮, 轮训练结果: 总损失Loss: {total_loss/total_samples:.5f}, Accuracy: {correct_samples/total_samples:.5f}, Time: {time.time()-start_time:.2f}s')

    # 6. 训练完成, 保存模型
    torch.save(model.state_dict(), './model/image_model.pth')

# 4. 模型测试
def evoluate(test_dataset):
    # 1. 创建数据加载器
    dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    # 2. 创建模型对象
    model = ImageModel()
    # 3. 加载模型参数
    model.load_state_dict(torch.load('./model/image_model.pth'))
    # 4. 定义统计变量, 预测的正确的样本数, 总样本数, 训练的开始时间
    correct_samples, total_samples, start_time = 0, 0, time.time()
    # 5. 遍历数据加载器, 获取当 每批次 的数据
    for x, y in dataloader:
        # 5.1 切换评估模式
        model.eval()
        # 5.2 模型预测
        y_pred = model(x)
        # 5.3 统计预测正确的样本数
        correct_samples += (torch.argmax(y_pred, dim=1) == y).sum()
        # 5.4 统计总样本数
        total_samples += x.shape[0]
    # 5.5 打印测试结果
    print(f'测试集准确率: {correct_samples/total_samples:.5f}, Time: {time.time()-start_time:.2f}s')


# 5. 测试
if __name__ == '__main__':
    # # 1. 获取数据集
    train_dataset, test_dataset = create_dataset()
    # print(f'训练集: {train_dataset.data.shape}')       # (50000, 32, 32, 3)
    # print(f'测试集: {test_dataset.data.shape}')        # (10000, 32, 32, 3)
    # #{'airplane': 0, 'automobile': 1, 'bird': 2, 'cat': 3, 'deer': 4, 'dog': 5, 'frog': 6, 'horse': 7, 'ship': 8, 'truck': 9}
    # print(f'数据集类别: {train_dataset.class_to_idx}')
    #
    # # 图像展示
    # plt.figure(figsize=(10, 10))
    # plt.imshow(train_dataset.data[11])
    # plt.title(train_dataset.targets[11])
    # plt.show()

    # # 2. 搭建神经网络
    # model = ImageModel()
    # # # 查看模型参数 参1: 网络模型 参2: 输入维度(CHW, 通道, 高, 宽)  参3: 批次大小
    # summary(model, (3, 32, 32),batch_size=BATCH_SIZE)

    # 3. 模型训练
    train(train_dataset)

    # 4. 模型测试
    evoluate(test_dataset)