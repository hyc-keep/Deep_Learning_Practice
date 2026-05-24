"""
案例:
    演示学习率衰减策略

学习率衰减策略介绍:
    目的:
        相比于AdaGrad, RMSProp, Adam方式, 我们可以通过 等间隔, 指定间隔, 指数等方式, 来手动控制学习率的调整

    分类:
        等间隔学习率衰减
        指定间隔学习率衰减
        指数学习率衰减

等间隔学习率衰减:
    step_size:  间隔的轮数, 即: 多少轮调整一次参数
    gamma:      学习率衰减系数, 即: lr新 = lr旧 * gamm

指定间隔学习率衰减:
    milestones = [50, 125, 160]:  定义的是第几轮调整学习率, 更符合梯度下降规律
    gamma:      学习率衰减系数, 即: lr新 = lr旧 * gamma

指数学习率衰减:
    前期学习率衰减快, 中期慢, 后期更慢
    公式:
        lr新 = lr旧 * gamma ** epoch

总结:
    等间隔学习率衰减:
        优点:
            直观, 易于测试, 适用于 大批量数据
        缺点:
            学习率变化较大, 可能跳过最优解
        应用场景:
            大型数据集, 较为简单的任务

    指定间隔学习率衰减:
        优点:
            易于调试, 稳定训练过程
        缺点:
            在某些情况下可能衰减过快, 导致优化提前停滞
        应用场景:
            对训练平稳性要求较高的任务

    指数学习率衰减:
        优点:
            平滑, 且考虑历史更新, 收敛稳定性较强
        缺点:
            超调调节较为复杂, 可能需要更多的资源
        应用场景:
            高精度训练, 避免过快收敛
"""

# 导包
import torch
from torch import optim
import matplotlib.pyplot as plt

# 定义函数, 演示等间隔学习率衰减
def dm01():
    # 1.定义变量, 记录初始的 学习率, 训练的轮数, 每轮训练的批次数
    lr, epochs, iteration = 0.1, 200, 10

    # 2.创建数据集, y_true, x, w
    # 真实值
    y_true = torch.tensor([0])
    # 输入特征
    x = torch.tensor([1.0],dtype=torch.float32)
    # 权重参数w, 需要自动微分(求导)
    w = torch.tensor([1.0],requires_grad= True,dtype=torch.float32)

    # 3.创建优化器对象, 动量法 --> 加速模型的收敛, 减少震荡
    # 参1: 待优化的权重参数, 参2: 学习率, 参3: 动量系数
    optimizer = optim.SGD([w],lr=lr,momentum= 0.9)

    # 4.创建学习率衰减对象
    # 思路1: 等间隔
    # 参1: 优化器对象, 参2: 间隔的轮数(多少轮停调整一次学习率), 参3: 学习率衰减系数 lr = lr * gamma
    scheduler = optim.lr_scheduler.StepLR(optimizer,step_size= 50,gamma= 0.5)

    # 5.创建两个列表,分别表示: 训练轮数和每轮训练的学习率
    epoch_list, lr_list = [],[]

    # 6.循环遍历训练轮数, 进行具体的训练
    for epoch in range(epochs):
        # 7.获取当前的轮数 和 学习率 ,并保存到列表中
        epoch_list.append(epoch)
        lr_list.append(scheduler.get_last_lr())  #获取最近的lr

        # 8.循环遍历, 每轮每批次进行训练
        for batch in range(iteration):
            # 9.计算预测值
            y_pred = x * w
            # 10.计算损失值,最小二乘法
            loss = ((y_pred - y_true) ** 2)
            # 11.梯度清零 + 反向传播 + 优化器更新参数
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        # 12.更新学习率
        scheduler.step()
    # 13.打印结果
    print(f'lr_list: {lr_list}')

    # 14.绘制学习率曲线
    plt.plot(epoch_list,lr_list)
    plt.xlabel("epoch")
    plt.ylabel("learning rata")
    plt.show()

# 定义函数, 指定间隔学习率衰减
def dm02():
    # 1.定义变量, 记录初始的 学习率, 训练的轮数, 每轮训练的批次数
    lr, epochs, iteration = 0.1, 200, 10

    # 2.创建数据集, y_true, x, w
    # 真实值
    y_true = torch.tensor([0])
    # 输入特征
    x = torch.tensor([1.0],dtype=torch.float32)
    # 权重参数w, 需要自动微分(求导)
    w = torch.tensor([1.0],requires_grad= True,dtype=torch.float32)

    # 3.创建优化器对象, 动量法 --> 加速模型的收敛, 减少震荡
    # 参1: 待优化的权重参数, 参2: 学习率, 参3: 动量系数
    optimizer = optim.SGD([w],lr=lr,momentum= 0.9)

    # 4.创建学习率衰减对象
    # 思路1: 等间隔
    # 参1: 优化器对象, 参2: 间隔的轮数(多少轮停调整一次学习率), 参3: 学习率衰减系数 lr = lr * gamma
    # scheduler = optim.lr_scheduler.StepLR(optimizer,step_size= 50,gamma= 0.5)

    # 思路2: 指定间隔
    milestones = [50, 125, 160]
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer,milestones=milestones,gamma= 0.5)

    # 5.创建两个列表,分别表示: 训练轮数和每轮训练的学习率
    epoch_list, lr_list = [],[]

    # 6.循环遍历训练轮数, 进行具体的训练
    for epoch in range(epochs):
        # 7.获取当前的轮数 和 学习率 ,并保存到列表中
        epoch_list.append(epoch)
        lr_list.append(scheduler.get_last_lr())  #获取最近的lr

        # 8.循环遍历, 每轮每批次进行训练
        for batch in range(iteration):
            # 9.计算预测值
            y_pred = x * w
            # 10.计算损失值,最小二乘法
            loss = ((y_pred - y_true) ** 2)
            # 11.梯度清零 + 反向传播 + 优化器更新参数
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        # 12.更新学习率
        scheduler.step()
    # 13.打印结果
    print(f'lr_list: {lr_list}')

    # 14.绘制学习率曲线
    plt.plot(epoch_list,lr_list)
    plt.xlabel("epoch")
    plt.ylabel("learning rata")
    plt.show()

# 定义函数, 指数学习率衰减
def dm03():
    # 1.定义变量, 记录初始的 学习率, 训练的轮数, 每轮训练的批次数
    lr, epochs, iteration = 0.1, 200, 10

    # 2.创建数据集, y_true, x, w
    # 真实值
    y_true = torch.tensor([0])
    # 输入特征
    x = torch.tensor([1.0],dtype=torch.float32)
    # 权重参数w, 需要自动微分(求导)
    w = torch.tensor([1.0],requires_grad= True,dtype=torch.float32)

    # 3.创建优化器对象, 动量法 --> 加速模型的收敛, 减少震荡
    # 参1: 待优化的权重参数, 参2: 学习率, 参3: 动量系数
    optimizer = optim.SGD([w],lr=lr,momentum= 0.9)

    # 4.创建学习率衰减对象
    # 思路1: 等间隔
    # 参1: 优化器对象, 参2: 间隔的轮数(多少轮停调整一次学习率), 参3: 学习率衰减系数 lr = lr * gamma
    # scheduler = optim.lr_scheduler.StepLR(optimizer,step_size= 50,gamma= 0.5)

    # 思路2: 指定间隔
    # milestones = [50, 125, 160]
    # scheduler = optim.lr_scheduler.MultiStepLR(optimizer,milestones=milestones,gamma= 0.5)

    # 思路3: 指数衰减
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer,gamma= 0.95)

    # 5.创建两个列表,分别表示: 训练轮数和每轮训练的学习率
    epoch_list, lr_list = [],[]

    # 6.循环遍历训练轮数, 进行具体的训练
    for epoch in range(epochs):
        # 7.获取当前的轮数 和 学习率 ,并保存到列表中
        epoch_list.append(epoch)
        lr_list.append(scheduler.get_last_lr())  #获取最近的lr

        # 8.循环遍历, 每轮每批次进行训练
        for batch in range(iteration):
            # 9.计算预测值
            y_pred = x * w
            # 10.计算损失值,最小二乘法
            loss = ((y_pred - y_true) ** 2)
            # 11.梯度清零 + 反向传播 + 优化器更新参数
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        # 12.更新学习率
        scheduler.step()
    # 13.打印结果
    print(f'lr_list: {lr_list}')

    # 14.绘制学习率曲线
    plt.plot(epoch_list,lr_list)
    plt.xlabel("epoch")
    plt.ylabel("learning rata")
    plt.show()

# 测试
if __name__ == '__main__':
    # dm01()
    # dm02()
    dm03()