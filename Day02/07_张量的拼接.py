"""
案例:
    演示张量的拼接

涉及到的API:
    cat()       不改变维度数,拼接张量.除了拼接的那个维度外,其他的维度数必须保持一致
    stack()     会改变维度数,拼接张量.所有的维度数都必须保持一致,可以在新维度拼接
"""
#导包
import torch

#设置随机数种子
torch.manual_seed(24)

#1.创建两个张量
t1 = torch.randint(1,10,size=(2,3))
print(f"t1:{t1},shape:{t1.shape}")

t2 = torch.randint(1,10,size=(2,3))
print(f"t2:{t2},shape:{t2.shape}")

#2.拼接张量 cat()
# t3 = torch.cat([t1,t2],dim=0)       #(2,3) + (2,3) = (4,3)
# print(f"t3:{t3},shape:{t3.shape}")

# t4 = torch.cat([t1,t2],dim=1)   #(2,3) + (2,3) = (2,6)
# print(f"t4:{t4},shape:{t4.shape}")

# t5 = torch.cat([t1,t2],dim=-1)   #(2,3) + (2,3) = (2,6)
# print(f"t4:{t4},shape:{t4.shape}")

# t6 = torch.cat([t1,t2],dim=2)   #报错
print("-" * 30)

#3.拼接张量 stack(),可以是任意维度,但是所有的维度数必须保持一致,在第几维拼接,就把2插入到第几维
t7 = torch.stack([t1,t2],dim=0)     #(2,3) + (2,3) = (2,2,3)
print(f"t7:{t7},shape:{t7.shape}")

t8 = torch.stack([t1,t2],dim=1)     #(2,3) + (2,3) = (2,2,3)
print(f"t8:{t8},shape:{t8.shape}")

t9 = torch.stack([t1,t2],dim=2)     #(2,3) + (2,3) = (2,3,2)
print(f"t9:{t9},shape:{t9.shape}")

# t10 = torch.stack([t1,t2],dim=3)    #报错,索引越界