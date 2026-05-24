"""
案例:
    演示张量的形状操作

涉及到的API:
    reshape()           在不改变张量内容的前提下(数据按顺序排),对其形状进行改变
    unsqueeze()         在指定的轴上增加(1)维度,等价于:升维
    squeeze()           删除所有为(1)的维度,等价于:降维
    transpose()         一次只能交换2个维度
    permute()           一次可以交换多个维度
    view()              只能修改连续张量的形状,连续张量 = 内存中存储顺序 和 在张量中显示的顺序相同
    contiguous()        把不连续的张量 --> 连续的张量 即:基于张量中显示的顺序,修改内存中的存储顺序
    is_contiguous()     判断张量是否是连续的

需要掌握的函数:
    reshape(),unsqueeze(),permute(),view()
"""

#导包
import torch
from numpy.ma.core import shape

#指定随机种子
torch.manual_seed(24)

#1.定义函数,演示:reshape()函数
def dm01():
    #1.定义2行3列的张量
    t1 = torch.randint(1,10,size=(2,3))
    print(f"t1: {t1},shape: {shape(t1)},row: {t1.shape[0]},column: {t1.shape[1]},{t1.shape[-1]}")

    #2.通过reshape()函数,将张量变成3行2列,和1行6列,和6行1列
    # t2 = t1.reshape(3,2)
    t2 = t1.reshape(1,6)
    t2 = t1.reshape(6,1)
    print(f"t2: {t2},shape: {shape(t2)},row: {t2.shape[0]},column: {t2.shape[1]},{t2.shape[-1]}")

    #3.通过reshape()函数,尝试将张量变成2行5列
    # t3 = t1.reshape(2,5)    #运行会报错,因为张量的元素个数必须等于新张量的元素个数

#2.定义函数,演示:unsqueeze(),squeeze()
def dm02():
    #1.定义2行3列的张量
    t1 = torch.randint(1,10,size=(2,3))
    print(f"t1:{t1},shape:{shape(t1)}")     #(2,3)

    #2.在0维添加一个维度
    t2 = t1.unsqueeze(0)
    print(f"t2:{t2},shape:{shape(t2)}")     #(1,2,3)

    #3.在1维添加一个维度
    t3 = t1.unsqueeze(1)
    print(f"t3:{t3},shape:{shape(t3)}")     #(2,1,3)

    #4.在2维添加一个维度
    t4 = t1.unsqueeze(2)
    print(f"t4:{t4},shape:{shape(t4)}")     #(2,3,1)

    #5.在3维添加一个维度
    # t5 = t1.unsqueeze(3)                  #运行会报错,因为张量的维度不能超过4维
    # print(f"t5:{t5},shape:{shape(t5)}")

    #6.删除所有为1的维度
    t6 = torch.randint(1,10,(2,1,3,1,1))
    print(f"t6:{t6},shape:{shape(t6)}")     #(2,1,3,1,1)

    t7 = t6.squeeze()
    print(f"t7:{t7},shape:{shape(t7)}")     #(2,3)

#3.定义函数,演示:transpose(),permute()
def dm03():
    #1,定义张量
    t1 = torch.randint(1,10,size=(2,3,4))
    print(f"t1:{t1},shape:{shape(t1)}")

    #2.交换0维和1维
    t2 = t1.transpose(0,1)
    print(f"t2:{t2},shape:{shape(t2)}")     #(3,2,4)

    #3.交换0维和2维
    # t3 = t1.transpose(0,2)
    t3 = t1.transpose(0,-1)
    print(f"t3:{t3},shape:{shape(t3)}")     #(4,3,2)

    #4.(2,3,4) --> (4,2,3)
    t4 = t1.permute(2,0,1)
    print(f"t4:{t4},shape:{shape(t4)}")

#4.定义函数,演示:view(),contiguous(),is_contiguous()
def dm04():
    #1.定义张量
    t1 = torch.randint(1,10,size=(2,3))
    print(f"t1:{t1},shape:{shape(t1)}")

    #2.判断张量是否是连续的,即:张量中显示的顺序,于内存中的存储顺序是否一致
    print(t1.is_contiguous())       # True

    #3.通过view()函数,修改张量的形状(2,3) --> (3,2)
    t2 = t1.view(3,2)
    print(f"t2:{t2},shape:{shape(t2)}")
    print(t2.is_contiguous())       # False

    #4.同过transpose()函数,交换维度
    t3 = t1.transpose(0,1)
    print(f"t3:{t3},shape:{shape(t3)}")     #(3,2)
    print(t3.is_contiguous())       # True

    #5.尝试把经过transpose()函数交换维度的t3张量,转换为(2,3)
    # t4 = t3.view(2,3)      #运行会报错,因为张量t3不是连续的

    #6.可以通过contiguous()函数,把t3张量转换为连续张量
    t5 = t3.contiguous().view(2,3)
    print(f"t5:{t5},shape:{shape(t5)}")
    print(t5.is_contiguous())

#5.测试
if __name__ == '__main__':
    # dm01()
    # dm02()
    # dm03()
    dm04()