""""
案例:
    演示detach()函数的功能,解决 自动微分的弊端

回顾:
    自动微分 = 求导,即: 基于损失函数,计算梯度
    结合权重更新公式: W新 = W旧 - 学习率 * 梯度, 来更新权重

问题:
    一个张量一旦设置了自动微分,这个张量就不能直接转成numpy的ndarray对象了,需要detach解决
"""
#导包
import torch
import numpy as np

#1.定义张量
#参1:数据,参2:是否自动微分,参3:数据类型
t1 = torch.tensor([10,20],requires_grad=True,dtype=torch.float32)
print(f"t1:{t1},类型:{type(t1)}")

#2.把t1张量转成numpy
# n1 = t1.numpy()     #报错,设置了自动微分,不能转成numpy
# print(f"n1:{n1},类型:{type(n1)}")

#3.通过detach()函数解决
t2 = t1.detach()
print(f"t2:{t2},类型:{type(t2)}")

#4.测试t1和t2是否共享同一个内存 --> 共享
t1.data[0] = 100
print(f"t1:{t1},类型:{type(t1)}")
print(f"t2:{t2},类型:{type(t2)}")

#5.查看t1和t2谁可以自动微分
print(f"t1:{t1.requires_grad}")     # True
print(f"t2:{t2.requires_grad}")     # False

#6.把t2转为 numpy对象
n2 = t2.numpy()
print(f"n2:{n2},类型:{type(n2)}")

#7.最终版
n3 = t1.detach().numpy()
print(f"n3:{n3},类型:{type(n3)}")