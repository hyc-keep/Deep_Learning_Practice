"""
案例:
    演示 张量 和 numpy 之间如何相互转换,以及如何从 标量张量 中提取内容

涉及到API:
    场景1:张量 --> numpy nd数组对象(ndarray)
    张量对象.numpy()            共享内存
    张量对象.numpy().copy()     不共享内存 链式编程

    场景2:numpy --> 张量
    from_numpy()                共享内存
    torch.tensor(nd数组)      不共享内存

    场景3:从 标量张量 中提取内容
    标量张量.item()

掌握:
    张量对象.numpy()    torch.tensor(nd数组)         标量张量.item()
"""
#导包
import torch
import numpy as np

#1.定义函数,演示: 张量 --> numpy
def dm01():
    #1.创建张量
    t1 = torch.tensor([1,2,3,4,5])
    print(f"t1:{t1},类型:{type(t1)}")

    #2.张量 --> numpy
    # n1 = t1.numpy()           #共享内存
    n1 = t1.numpy().copy()      #不共享内存
    print(f"n1:{n1},类型:{type(n1)}")

    #3.演示上述方式,是否共享内存
    n1[0] = 100
    print(f"n1:{n1},类型:{type(n1)}")     #[100,2,3,4,5]
    print(f"t1:{t1},类型:{type(t1)}")     #[?,2,3,4,5]


#2.定义函数,演示: numpy --> 张量
def dm02():
    #1.创建一个numpy数组
    n1 = np.array([11,22,33])
    print(f"n1:{n1},类型:{type(n1)}")

    #2.numpy --> 张量
    # t1 = torch.from_numpy(n1).type(torch.int32)     #共享内存
    t1 = torch.tensor(n1)                           #不共享内存
    print(f"t1:{t1},类型:{type(t1)}")

    #3.演示上述方式,是否共享内存
    n1[0] = 100
    print(f"n1:{n1},类型:{type(n1)}")     #[100,22,33]
    print(f"t1:{t1},类型:{type(t1)}")     #[?,22,33]

#3.定义函数,演示: 从 标量张量(只有一个数值的张量) 中提取内容
def dm03():
    #1.创建一个标量张量
    t1 = torch.tensor(100)          #可以转换
    t1 = torch.tenspr([100,])       #可以转换
    # t1 = torch.tensor([100,200])    #不可以转换(超过1个值)
    print(f"t1:{t1},类型:{type(t1)}")

    #2.将标量张量,转换成标量
    a = t1.item()
    print(f"a:{a},类型:{type(a)}")

#测试函数
if __name__ == '__main__':
    # dm01()
    # dm02()
    dm03()