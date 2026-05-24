"""
案例:
    演示自动微分模块,循环实现 计算梯度,更新参数

需求:
    loss = w**2 + 20 的极小值点,并打印loss是最小值时 w的值(梯度)

操作步骤:
    1.定义点 w=10, required_grad = True,dtype= torch.float32
    2.定义函数 loss = w**2 + 20
    3.利用梯度下降法,循环迭代100次,求最优解
    3.1正向计算(前向传播)
    3.2梯度清零 w.grad.zero_()
    3.3反向传播
    3.4梯度更新 w.data = w.data - 0.01 * w.grad
"""
#导包
import torch

# 1.定义点w = 10, required_grad = True, dtype = torch.float32
#参一：初始值,参二：自动微分,参三：数据类型,浮点型
w = torch.tensor(10,requires_grad=True,dtype=torch.float32)

# 2.定义函数loss = w ** 2 + 20
loss = w ** 2 + 20      #求导: loss' = 2W

# 3.利用梯度下降法, 循环迭代100次, 求最优解
print(f"开始 权重值:{w}, (0.01 * w.grad): 无, 损失值loss:{loss}")

for i in range(1,101):
    # 3.1正向计算(前向传播)
    loss = w ** 2 + 20

    # 3.2梯度清零w.grad.zero_()     默认梯度会 accumulate,即梯度会累加,所以每次迭代时,需要清零
    #当第一次迭代时,w.grad为None,所以需要做非空判断
    if w.grad is not None:
        w.grad.zero_()

    # 3.3 反向传播
    loss.sum().backward()

    # 3.4 梯度更新 w.data = w.data - 0.01 * w.grad
    print(f"梯度值为:{w.grad}")
    w.data = w.data - 0.01 * w.grad

    #3.5打印本次梯度更新后的权重值,损失值
    print(f"第{i}次 梯度更新后 权重值:{w}, (0.01 * w.grad): {0.01 * w.grad:.5f}, 损失值loss:{loss:.5f}")

#4.打印最终结果
print(f"最终结果 权重值:{w}, 梯度值:{w.grad:.5f},损失值loss:{loss:.5f}")