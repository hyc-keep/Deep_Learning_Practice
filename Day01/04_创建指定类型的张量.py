"""
案例:
    创建指定类型的张量

涉及到的函数:
    type(torch支持的数据类型)
    half()/float()/double()/short()/int()/long()

需要掌握:
     type(torch支持的数据类型)
"""
#导包
import torch

#场景1:直接创建指定类型的张量
t1 = torch.tensor([1,2,3,4,5],dtype=torch.float)    #默认:float32
print(f"t1:{t1},(元素)类型:{t1.dtype},(张量)类型:{type(t1)}")
print("-" * 30)

#场景2:创建好张量后 -->做类型转换
#思路1:使用type(),推荐
t2 = t1.type(torch.int32)
print(f"t2:{t2},(元素)类型:{t2.dtype},(张量)类型:{type(t2)}")
print("-" * 30)

#思路2:使用half()/float()/double()/short()/int()/long()
print(t2.half())
print(t2.float())
print(t2.double())
print(t2.short())
print(t2.int())
print(t2.long())
