"""
梯度下降相关介绍:
    概述:
        梯度下降是结合 本次损失函数的导数(作为梯度) 基于学习率 来更新权重的
    公式:
    todo    W新 = W旧 - 学习率 * Gt(本次计算的梯度)
    存在的问题:
        1. 遇到平缓区域, 梯度下降(权重更新)可能会慢,因为梯度Gt小
        2. 可能会遇到 鞍点(梯度为0)
        3. 可能会遇到 局部最小值(梯度为0)
    解决思路:
        从上述的 学习率 或 梯度 入手,进行优化,于是有了:
            动量法 Momentum
            自适应学习率 AdaGrad
            RMSProp
            综合衡量 Adam

    动量法 Momentum:
        动量法公式:
        todo    St = β * St-1 + (1 - β) * Gt
        解释:
            St:     本次的指数加权平均梯度
            β:      调节权重系数, β越大, 数据越平缓, 历史指数加权平均的 比重越大, 本次梯度的 比重越小
            St-1:   历史的指数加权平均梯度
            Gt:     本次计算的梯度
        加入动量法后的 梯度下降公式:
        todo  W新 = W旧 - 学习率 * St

    自适应学习率 AdaGrad:
        公式:
        todo    St = St-1 + Gt * Gt(本次计算的梯度)
        解释:
            St:     累计平方梯度
            St-1:   历史累计平方梯度
            Gt:     本次计算的梯度
        学习率:
            学习率 = 学习率 / ((Sqrt(St) + 小常数 )      小常数: 1e-10, 目的是防止分母为0
        梯度下降公式:
        todo    W新 = W旧 - 调整后的学习率 * Gt(本次计算的梯度)
        缺点:
            可能会导致学习率过早,过量的降低,导致模型后期学习率太小,较难找到最优解

    自适应学习率 RMSProp:
        公式:
        todo    St = β * St-1 + (1 - β) * Gt * Gt(本次计算的梯度)
        解释:
            St:     累计平方梯度
            St-1:   历史累计平方梯度
            Gt:     本次计算的梯度
            β:      调和权重系数
        学习率:
            学习率 = 学习率 / ((Sqrt(St) + 小常数 )      小常数: 1e-10, 目的是防止分母为0
        梯度下降公式:
        todo    W新 = W旧 - 调整后的学习率 * Gt(本次计算的梯度)
        优点:
            RMSProp通过引入 调和权重系数β, 控制历史梯度 对 历史梯度信息获取的多少

    自适应矩估计:Adam
    思路:
        既优化学习率,又优化梯度
    公式:
        一阶矩: 算均值
            Mt = β1 * Mt-1 + (1 - β1) * Gt          充当: 梯度      Momentum
            St = β2 * St-1 + (1 - β2) * Gt * Gt     充当: 学习率     RMSProm
        二阶矩: 算梯度的方差
            Mt^ = Mt / (1 - β1 ^ t)
            St^ = St / (1 - β2 ^ t)
        梯度下降公式:
        todo    W新 = W旧 - 学习率 / (sqrt(St^) + 小常数)  + Mt^    即: Adam = Momentum + RMSProm

    总结: 如何选择梯度下降优化方法?
        简单任务和较小的模型:
            SGD,动量法Momentum
        复制任务或有大量数据:
            Adam
        需要处理稀疏数据或文本数据:
            AdaGrad,RMSProp
"""

# 导包
import torch
import torch.nn as nn
import torch.optim as optim

# 1.动量法
def dm01_momentum():
    # 1.初始化权重参数
    w = torch.tensor([1.0],requires_grad=True,dtype=torch.float)
    # 2.定义损失函数
    criterion = ((w ** 2) / 2.0)
    # 3. 创建优化器 --> 基于SGD(随机梯度下降), 加入参数 Momentum , 就是动量法
    optimizer = optim.SGD(params=[w],lr=0.01,momentum=0.9)      # momentum=0(默认),只考虑本次梯度
    # 4.计算梯度值: 梯度清零 + 反向传播 + 更新参数
    optimizer.zero_grad()
    criterion.sum().backward()
    optimizer.step()
    print(f'w:{w},w.grad:{w.grad}')

    # 5重复上述步骤 第2次,更新权重参数
    # 5.1 定义损失函数
    criterion = ((w ** 2) / 2.0)
    # 5.2计算梯度值: 梯度清零 + 反向传播 + 更新参数
    optimizer.zero_grad()
    criterion.sum().backward()
    optimizer.step()
    print(f'w:{w},w.grad:{w.grad}')

# 2.自适应学习率
def dm02_AdaGrad():
    # 1.初始化权重参数
    w = torch.tensor([1.0], requires_grad=True, dtype=torch.float)
    # 2.定义损失函数
    criterion = ((w ** 2) / 2.0)

    # 3. 创建优化器
    # 思路1:  基于SGD(随机梯度下降), 加入参数 Momentum , 就是动量法
    # optimizer = optim.SGD(params=[w], lr=0.01, momentum=0.9)  # momentum=0(默认),只考虑本次梯度

    # 思路2:  基于AdaGrad(自适应学习率)
    optimizer = optim.Adagrad(params=[w], lr=0.01)

    # 4.计算梯度值: 梯度清零 + 反向传播 + 更新参数
    optimizer.zero_grad()
    criterion.sum().backward()
    optimizer.step()
    print(f'w:{w},w.grad:{w.grad}')

    # 5重复上述步骤 第2次,更新权重参数
    # 5.1 定义损失函数
    criterion = ((w ** 2) / 2.0)
    # 5.2计算梯度值: 梯度清零 + 反向传播 + 更新参数
    optimizer.zero_grad()
    criterion.sum().backward()
    optimizer.step()
    print(f'w:{w},w.grad:{w.grad}')

# 3.RMSProp:
def dm03_RMSProp():
    # 1.初始化权重参数
    w = torch.tensor([1.0], requires_grad=True, dtype=torch.float)
    # 2.定义损失函数
    criterion = ((w ** 2) / 2.0)

    # 3. 创建优化器
    # 思路1:  基于SGD(随机梯度下降), 加入参数 Momentum , 就是动量法
    # optimizer = optim.SGD(params=[w], lr=0.01, momentum=0.9)  # momentum=0(默认),只考虑本次梯度

    # 思路2:  基于AdaGrad(自适应学习率)
    # optimizer = optim.Adagrad(params=[w], lr=0.01)

    # 思路3:  基于RMSProp(自适应学习率)
    optimizer = optim.RMSprop(params=[w], lr=0.01, alpha=0.99)

    # 4.计算梯度值: 梯度清零 + 反向传播 + 更新参数
    optimizer.zero_grad()
    criterion.sum().backward()
    optimizer.step()
    print(f'w:{w},w.grad:{w.grad}')

    # 5重复上述步骤 第2次,更新权重参数
    # 5.1 定义损失函数
    criterion = ((w ** 2) / 2.0)
    # 5.2计算梯度值: 梯度清零 + 反向传播 + 更新参数
    optimizer.zero_grad()
    criterion.sum().backward()
    optimizer.step()
    print(f'w:{w},w.grad:{w.grad}')

# 4.Adam
def dm04_Adam():
    # 1.初始化权重参数
    w = torch.tensor([1.0], requires_grad=True, dtype=torch.float)
    # 2.定义损失函数
    criterion = ((w ** 2) / 2.0)

    # 3. 创建优化器
    # 思路1:  基于SGD(随机梯度下降), 加入参数 Momentum , 就是动量法
    # optimizer = optim.SGD(params=[w], lr=0.01, momentum=0.9)  # momentum=0(默认),只考虑本次梯度

    # 思路2:  基于AdaGrad(自适应学习率)
    # optimizer = optim.Adagrad(params=[w], lr=0.01)

    # 思路3:  基于RMSProp(自适应学习率)
    # optimizer = optim.RMSprop(params=[w], lr=0.01, alpha=0.99)

    # 思路4:  基于Adam(自适应矩估计)
    optimizer = optim.Adam(params=[w], lr=0.01, betas=(0.9, 0.999))     # todo betas=(梯度用的, 学习率用的)

    # 4.计算梯度值: 梯度清零 + 反向传播 + 更新参数
    optimizer.zero_grad()
    criterion.sum().backward()
    optimizer.step()
    print(f'w:{w},w.grad:{w.grad}')

    # 5重复上述步骤 第2次,更新权重参数
    # 5.1 定义损失函数
    criterion = ((w ** 2) / 2.0)
    # 5.2计算梯度值: 梯度清零 + 反向传播 + 更新参数
    optimizer.zero_grad()
    criterion.sum().backward()
    optimizer.step()
    print(f'w:{w},w.grad:{w.grad}')

# 测试
if __name__ == '__main__':
    # dm01_momentum()
    # dm02_AdaGrad()
    # dm03_RMSProp()
    dm04_Adam()