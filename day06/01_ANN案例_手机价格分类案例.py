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

优化思路:
    1. 优化方法从SGD --> Adam            √
    2. 学习率从0.001 --> 0.0001         √
    3. 对数据进行标准化                  √
    4. 增加网络的深度, 每层的神经元数量
    5. 调整训练的轮数
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
from torchsummary import summary                    # 查看模型结构
from sklearn.preprocessing import StandardScaler    # 标准化

# todo 1. 定义函数, 构建数据集
def creat_dataset():
    # 1.加载csv文件数据集
    data = pd.read_csv('./data/手机价格预测.csv')

    # 2.获取x特征列 和 y标签列 得到的是 Pandas.Series类型    把 x 和 y 转换为数组类型
    x = data.iloc[:,:-1].values
    y = data.iloc[:,-1].values

    # 3.把特征列转出浮点数
    x = x.astype(np.float32)
    y = y.astype(np.int64)

    # 4.切分训练集和测试集
    # 参数1: x特征 参数2: y标签 参数3: 测试集比例 参数4: 随机种子 参数5: 样本的分布是否相同(即:参考y的类别进行抽取数据)
    x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2,random_state=3,stratify=y)

    # 优化: 将数据集进行标准化
    transfer = StandardScaler()
    x_train = transfer.fit_transform(x_train)
    x_test = transfer.transform(x_test)

    # 5.把数据集转换成张量 思路: 数据 --> 张量 --> 数据集对象 --> 数据加载器
    train_dataset = TensorDataset(torch.tensor(x_train), torch.tensor(y_train))
    test_dataset = TensorDataset(torch.tensor(x_test), torch.tensor(y_test))

    # 6.返回                               20(充当输入特征数)   4(充当输出标签数)
    return train_dataset, test_dataset, x_train.shape[1],len(np.unique(y))

# todo 2. 搭建神经网络
class PhonePriceModel(nn.Module):
    # 1.在init魔法方法中, 初始化父类成员, 以及搭建神经网络
    def __init__(self, input_dim, output_dim):      #输入: 20 输出: 4
        # 1.1初始化父类成员
        super(PhonePriceModel,self).__init__()
        # 1.2搭建神经网络
        # 隐藏层1
        self.linear1 = nn.Linear(input_dim, 128)
        # 隐藏层2
        self.linear2 = nn.Linear(128, 256)
        # 输出层
        self.output = nn.Linear(256, output_dim)

    # 2.定义前向传播方法 forward()
    def forward(self,x):
        # 2.1 隐藏层1: 加权求和 + 激活函数relu
        x = torch.relu(self.linear1(x))
        # 2.2 隐藏层2: 加权求和 + 激活函数relu
        x = torch.relu(self.linear2(x))
        # 2.3 输出层: 加权求和 + 激活函数softmax --> 这里只需要做 加权求和 正常写法是需要softmax,但是后面用 CrossEntropyLoss() 代替
        # x = torch.softmax(self.output(x), dim=1)
        # CrossEntropyLoss() = softmax + 损失计算
        x = self.output(x)
        # 2.4返回处理结果
        return x


# todo 3. 模型训练
def train(train_dataset,input_dim,output_dim):
    # 1.创建数据加载器, 流程: 数据集对象 --> 数据加载器
    # 参1: 数据集对象(1600) 参2: 每批次的数据条数 参3: 是否打乱数据(训练集打乱, 测试集不打乱)
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    # 2.创建神经网络模型
    model = PhonePriceModel(input_dim, output_dim)
    # 3.定义损失函数, 因为是多分类, 所以使用交叉熵损失函数 CrossEntropyLoss()
    criterion = nn.CrossEntropyLoss()
    # 4.创建优化器对象
    optimizer = optim.Adam(model.parameters(), lr=0.0001)
    # 5.模型训练
    # 5.1定义变量, 记录训练的 总轮数
    epochs = 50
    # 5.2开始训练
    for epoch in range(epochs):
        # 5.2.1 定义变量, 记录累加每批次的总损失, 累加每批次的实际样本数量
        total_loss, total_sample = 0.0, 0
        # 5.2.2 定义变量, 表示训练开始的时间
        start = time.time()
        # 5.2.3 开始本轮的 各个批次的训练
        for x, y in train_loader:
            # 5.2.4 切换模型状态 训练模式
            model.train()
            # 5.2.5 模型预测
            y_pred = model(x)
            # 5.2.6 计算损失
            loss = criterion(y_pred, y)
            # 5.2.7 梯度清零 + 反向传播 + 优化器更新参数
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # 5.2.8累加损失值
            total_loss += loss.item() * x.shape[0]      # 累加每批次所有样本的损失值
            total_sample += x.shape[0]                  # 累加每批次的实际样本数量  1600
        # 5.2.4 至此, 本轮训练结束, 打印训练信息
        print(f'第{epoch+1}轮训练结束, 损失值: {total_loss / total_sample:.4f}, 训练时间: {time.time() - start:.4f}s')

    # 6.至此,说明训练结束, 保存模型参数(权重w ,偏置b)
    # 参1: 模型对象的参数(权重w ,偏置b)     参2: 保存模型参数的文件名(路径)
    # print(f'\n\n模型参数的信息: {model.state_dict()}\n\n')
    torch.save(model.state_dict(), './model/phone_price_model.pth')     # 后缀名用: .pth .plk .pickle均可


# todo 4. 模型测试
def evaluate(test_dataset,input_dim,output_dim):
    # 1.创建神经网络模型
    model = PhonePriceModel(input_dim, output_dim)
    # 2.加载模型参数
    model.load_state_dict(torch.load('./model/phone_price_model.pth'))
    # 3.创建测试集的 数据加载器
    # 参1: 数据集对象(400) 参2: 每批次的数据条数 参3: 是否打乱数据(训练集打乱, 测试集不打乱)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)
    # 4.定义变量, 记录预测正确的样本个数
    correct = 0
    # 5.从数据加载器中, 获取测试集的样本数据
    for x, y in test_loader:
        # 5.1切换模型状态 测试模式
        model.eval()
        # 5.2模型预测
        y_pred = model(x)                #因为在输出层中,没有softmax(),所以这里输出的不是类别,而是类别得分
        # print(f'y_pred: {y_pred}')      # [[0分类的得分,1分类的得分,2分类的得分,3分类的得分], [] , [] , ...]

        # 5.3 根据加权求和, 得到类别, 用argmax()获取最大值对应的索引, 就是类别
        y_pred = torch.argmax(y_pred, dim=1)    #dim=1 表示对每行进行操作
        # print(f'y_pred: {y_pred}')        # [第1条数据的预测分类,第2条数据的预测分类,第3条数据的预测分类,第4条数据的预测分类]

        # 5.4 统计预测正确的样本个数
        # print(y_pred == y)              #tensor([True, True, True, True, False, ...]) True=1,False=0
        # print(y_pred == y).sum()        # tensor(样本正确数量)    n True(1) + m False(0) = 正确数量   m + n = 400
        correct += (y_pred == y).sum()

    # 6.至此, 模型测试结束, 打印测试结果
    print(f'准确率(Accurate): {correct / len(test_dataset):.4f}')

if __name__ == '__main__':
    # 1.准备数据集
    train_dataset, test_dataset, input_dim, output_dim = creat_dataset()
    # print(f'训练集的数据集对象: {train_dataset}')
    # print(f'测试集的数据集对象: {test_dataset}')
    # print(f'输入特征维度: {input_dim}')         20
    # print(f'输出标签维度: {output_dim}')         4

    # 2.构建神经网络模型
    # model = PhonePriceModel(input_dim, output_dim)
    # 计算模型参数
    # 参1: (神经网络)模型对象 参2: 输入数据维度(16表示批次大小,输入特征数20)
    # summary(model,input_size=(16,input_dim))

    # 3.模型训练
    train(train_dataset,input_dim,output_dim)

    # 4.模型测试
    evaluate(test_dataset,input_dim,output_dim)