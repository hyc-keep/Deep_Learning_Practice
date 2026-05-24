"""
案例:
    演示张量的常用运算函数

涉及到的API(函数):
    sum(),max(),min(),mean()                    --> 都有dim参数,0代表列,1代表行
    pow().sqrt(),exp(),log(),log2(),log10()     -->都没有dim参数

掌握:
    sum(),max(),min(),mean()
    a ** 2  次幂
"""
#导包
import torch

#1.定义张量,记录初值
t1 = torch.tensor([
    [1,2,3],
    [4,5,6]
],dtype=torch.float)
print("t1:", t1)

#2.演示有dim参数的运算
#sum() 求和
print(t1.sum(dim=0))    #列求和
print(t1.sum(dim=1))    #行求和
print(t1.sum())          #全部求和
print("-" * 30)

#max() 最大值
print(t1.max(dim=0))    #列最大值
print(t1.max(dim=1))    #行最大值
print(t1.max())         #全部最大值
print("-" * 30)

#min() 最小值
print(t1.min(dim=0))    #列最小值
print(t1.min(dim=1))    #行最小值
print(t1.min())         #全部最小值
print("-" * 30)

#mean() 平均值
print(t1.mean(dim=0))   #列平均值
print(t1.mean(dim=1))   #行平均值
print(t1.mean())        #全部平均值
print("*" * 30)

#3.演示无dim参数的运算
#pow() 幂次方
print(t1.pow(2))    #矩阵的元素求2次方
print(t1 ** 2)      #效果同上

#sqrt() 平方根
print(t1.sqrt())    #矩阵的元素求平方根

#exp() 矩阵的元素求e的幂次
print(t1.exp())     #矩阵的元素求e的幂次 e^1,e^2,e^3,e^4,e^5,e^6

#log() 矩阵的元素求对数
print(t1.log())     #以e为底,矩阵的元素求对数

#log2() 矩阵的元素求对数
print(t1.log2())    #以2为底,矩阵的元素求对数

#log10() 矩阵的元素求对数
print(t1.log10())   #以10为底,矩阵的元素求对数