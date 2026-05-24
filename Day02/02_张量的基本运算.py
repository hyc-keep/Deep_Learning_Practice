"""
案例:
    演示张量的基本运算

涉及到的API:
    add(),sub(),mul(),div(),neg()   --> 加减乘除,取反,add,substract,multiply,divide,
    add_(),sub_(),mul_(),div_(),neg_()  --> 功能同上,只不过可以修改源数据,类似pandas部分的inplace=True

需要掌握的:
    1.可以用 +,-,*,/ 符号 来代替上述的加减乘除函数
    2.如果张量是和数值进行运算,则 该数值会和张量中的每一个元素进行运算
"""
#导包
import torch

#1.创建张量
t1 = torch.tensor([1,2,3])

#2.张量加法
# t2 = t1.add(10)     #不修改源数据
# t2 = t1 + 10        #效果同上

# t1.add_(10)    #修改源数据
# t1 += 10     #效果同上

#其他运算
# t2 = t1.sub(10)
# t2 = t1 - 10
#
# t2 = t1.mul(10)
# t2 = t1 * 10
#
# t2 = t1.div(10)     #有浮点数
# t2 = t1 / 10
#
# t2 = t1.neg()
# t2 = -t1


#3.打印结果
print(f"t1:{t1},类型:{type(t1)}")
print(f"t2:{t2},类型:{type(t2)}")