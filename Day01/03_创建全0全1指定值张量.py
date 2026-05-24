"""
案例:
    演示如何创建全0,全1,指定值的张量

涉及到函数如下:
    torch.ones 和 torch.ones_like 创建全1张量
    torch.zeros 和torch.zeros_like 创建全0张量
    torch.full 和 torch.full_like 创建全指定张量

需要掌握的函数:
    zroes(),偏置b全为0(线性),full()
"""
#导包
import torch

#1.定义函数,演示: torch.ones 和 torch.ones_like 创建全1张量
def dm01():
    #场景1:torch.ones 和 torch.ones_like 创建全1张量
    t1 = torch.ones(size=(2,3))     #创建2行3列全1张量
    print(f"t1:{t1},type:{type(t1)}")
    print("-" * 30)

    #3行2列
    t2 = torch.tensor([[1,2],[3,4],[5,6]])
    print(f"t2:{t2},type:{type(t2)}")

    #t3 --> 基于t2的形状创建全1张量
    t3 = torch.ones_like(t2)
    print(t3)
    print("-" * 30)

    #场景2: torch.zeros 和torch.zeros_like 创建全0张量
    t4 = torch.zeros(size=(2,3))    #创建2行3列全0张量
    print(f"t4:{t4},type:{type(t4)}")

    #3行2列
    t5 = torch.tensor([[1,2],[3,4],[5,6]])
    print(f"t5:{t5},type:{type(t5)}")

    #t6 --> 基于t5的形状创建全0张量
    t6 = torch.zeros_like(t5)
    print(f"t6:{t6},type:{type(t6)}")

    #场景3: torch.full 和 torch.full_like 创建全指定张量
    t7 =  torch.full(size=(2,3),fill_value=255)
    print(f"t7:{t7},type:{type(t7)}")
    print("-" * 30)

    #3行2列
    t8 = torch.tensor([[1,2],[3,4],[5,6]])
    print(f"t8:{t8},type:{type(t8)}")
    print("-" * 30)

    #t9 --> 基于t8的形状创建全指定张量
    t9 = torch.full_like(t8,fill_value=255)
    print(f"t9:{t9},type:{type(t9)}")


#测试函数
if __name__ == "__main__":
    dm01()
