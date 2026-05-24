"""
案例:
    演示张量的点乘和矩阵的乘法

点乘:
    要求:两个张量的维度要保持一致,对应元素直接做相应的操作
    API:
        t1 * t2
        t1.mul(t2)          #multiply乘法

矩阵乘法:
    要求:两个张量,第一个张量的列等于第二个张量的行(A列B行)
    结果:A行B列
    API:
        t1 @ t2
        t1.matmul(t2)
        t1.dot(t2)      #只对一维有效
"""
#导包
import torch

#1.定义函数,演示张量的点乘 和矩阵
def dm01():
    #1.定义张量
    t1 = torch.tensor([[1,2,3],[4,5,6]])
    print(f"t1:{t1}")

    #2.定义张量
    t2 = torch.tensor([[1,2,3],[4,5,6]])
    print(f"t2:{t2}")

    #3.点乘
    # t3 = t1 * t2
    t3 = t1.mul(t2)
    print(f"t3:{t3}")

#2.定义函数,演示矩阵的乘法
def dm02():
    #1.定义张量,2行3列
    t1 = torch.tensor([[1,2,3],[4,5,6]])
    print(f"t1:{t1}")

    #2.定义张量,3行2列
    t2 = torch.tensor([[1,2],[3,4],[5,6]])
    print(f"t2:{t2}")

    #3.矩阵乘法
    # t3 = t1 @ t2
    t3 = t1.matmul(t2)
    print(f"t3:{t3}")

#3.测试函数
if __name__ == '__main__':
    # dm01()
    dm02()