"""
案例:
    演示张量的基本创建方式

张量:
    Pytorch框架属于最常用的深度学习框架,无论是ANN(人工神经网络),CNN(卷积神经网络),RNN(循环神经网络)
    底层在处理数据时,都是使用张量来处理的

    张量 --> 存储同一类型元素的容器,且元素值必须是数值才行.

张量的基本创建方式:
    torch.tensor 根据指定数据创建张量
    torch.Tensor 根据形状创建张量,也可以用来创建指定数据的张量
    torch.IntTensor,torch.FloatTensor,torch.DoubleTensor 创建指定类型的张量

细节:
    Tensor方式较与tensor方式,可以基于形状来直接创建张量

掌握:
    tensor(值,类型):
        data = [[1,2,3],[4,5,6]]
        t2 = torch.tensor(data)
"""

#导包
import torch
import numpy

#1.定义函数,演示: torch.tensor 根据指定数据创建张量
def dm01():     #掌握torch.tensor 用的多
    #场景1 标量 张量
    t1 = torch.tensor(10)
    print(f"t1:{t1},type:{type(t1)}")
    print("-" * 30)

    #场景2 二维列表 --> 张量
    data = [[1,2,3],[4,5,6]]
    t2 = torch.tensor(data)
    print(f"t2:{t2},type:{type(t2)}")
    print("-" * 30)

    #场景3: numpy nd数组 --> 张量
    data = numpy.random.randint(0,10,(2,3))
    t3 = torch.tensor(data,dtype=torch.float)
    print(f"t3:{t3},type:{type(t3)}")
    print("-" * 30)

    #场景4: 尝试直接创建指定维度维度的张量(如2行3列)(torch.tensor不包含形状)
    # t4 = torch.tensor(2,3)      #报错
    # print(f"t4:{t4},type:{type(t4)}")

#2.定义函数,演示: torch.Tensor 根据形状创建张量,也可以用来创建指定数据的张量
def dm02():
    #场景1 标量 张量
    t1 = torch.Tensor(10)
    print(f"t1:{t1},type:{type(t1)}")
    print("-" * 30)

    #场景2 二维列表 --> 张量
    data = [[1,2,3],[4,5,6]]
    t2 = torch.Tensor(data)
    print(f"t2:{t2},type:{type(t2)}")
    print("-" * 30)

    #场景3: numpy nd数组 --> 张量
    data = numpy.random.randint(0,10,(2,3))
    t3 = torch.Tensor(data)
    print(f"t3:{t3},type:{type(t3)}")
    print("-" * 30)

    # #场景4: 尝试直接创建指定维度维度的张量(如2行3列)
    t4 = torch.Tensor(2,3)
    print(f"t4:{t4},type:{type(t4)}")

#3.定义函数,演示: torch.IntTensor,torch.FloatTensor,torch.DoubleTensor 创建指定类型的张量
def dm03():
    #场景1 标量 张量
    t1 = torch.IntTensor(10)
    print(f"t1:{t1},type:{type(t1)}")
    print("-" * 30)

    #场景2 二维列表 --> 张量
    data = [[1,2,3],[4,5,6]]
    t2 = torch.IntTensor(data)
    print(f"t2:{t2},type:{type(t2)}")
    print("-" * 30)

    #场景3: numpy nd数组 --> 张量
    data = numpy.random.randint(0,10,(2,3))
    t3 = torch.IntTensor(data)
    print(f"t3:{t3},type:{type(t3)}")
    print("-" * 30)

    #场景4:如果类型不匹配 ,会尝试自动转换类型
    data = numpy.random.randint(0, 10, (2, 3))
    t4 = torch.FloatTensor(data)        #默认:float32
    print(f"t4:{t4},type:{type(t4)}")
    print("-" * 30)

#4.测试函数
if __name__ == "__main__":
    dm01()    #掌握
    # dm02()
    # dm03()