"""
案例:
    演示参数初始化的7种方式

参数初始化的目的:
    1. 防止梯度消失 或 爆炸
    2. 提高收敛速度
    3. 打破对称性

参数初始化的方式:
    无法打破对称性的:
        全0,全1,固定值
    可以打破对称性的:
        随机初始化,正态分布初始化,kaimin初始化,xavier初始化

总结:
    1. 记忆kaiming初始化,xavier初始化,全0初始化
    2. 关于初始化的选择:
        激活函数ReLU及其系列:   优先用kaiming初始化
        激活函数非ReLU:        优先用xavier
        如果是浅层网络:        可以考虑使用 随机初始化
"""
import torch.nn as nn   #nn 神经网络

# 1.均匀分布随机初始化
def dm01():
    # setp1: 创建线性层,输入5个神经元,输出3个神经元
    Linear = nn.Linear(5,3)
    # step2: 对权重(w)进行随机初始化,从0-1均匀分布产生参数
    nn.init.uniform_(Linear.weight)
    # step3: 对偏置(b)进行随机初始化,从0-1均匀分布产生参数
    nn.init.uniform_(Linear.bias)
    # step4: 打印生成结果
    print(Linear.weight.data)
    print(Linear.bias.data)

# 2.固定初始化
def dm02():
    # setp1: 创建线性层,输入5个神经元,输出3个神经元
    Linear = nn.Linear(5,3)
    # step2: 对权重(w)进行固定初始化,固定产生参数
    nn.init.constant_(Linear.weight,3)
    # step3: 对偏置(b)进行固定初始化,固定产生参数
    nn.init.constant_(Linear.bias,3)
    # step4: 打印生成结果
    print(Linear.weight.data)
    print(Linear.bias.data)

# 3.全0初始化
def dm03():
    # setp1: 创建线性层,输入5个神经元,输出3个神经元
    Linear = nn.Linear(5,3)
    # step2: 对权重(w)进行全0初始化,产生参数0
    nn.init.zeros_(Linear.weight)
    # step3: 对偏置(b)进行全0初始化,产生参数0
    nn.init.zeros_(Linear.bias)
    # step4: 打印生成结果
    print(Linear.weight.data)
    print(Linear.bias.data)

# 4.全1初始化
def dm04():
    # setp1: 创建线性层,输入5个神经元,输出3个神经元
    Linear = nn.Linear(5,3)
    # step2: 对权重(w)进行全1初始化,产生参数1
    nn.init.ones_(Linear.weight)
    # step3: 对偏置(b)进行全1初始化,产生参数1
    nn.init.ones_(Linear.bias)
    # step4: 打印生成结果
    print(Linear.weight.data)
    print(Linear.bias.data)

# 5.正态分布随机初始化
def dm05():
    # setp1: 创建线性层,输入5个神经元,输出3个神经元
    Linear = nn.Linear(5,3)
    # step2: 对权重(w)进行正态分布随机初始化,产生参数
    nn.init.normal_(Linear.weight)
    # step3: 对偏置(b)进行正态分布随机初始化,产生参数
    nn.init.normal_(Linear.bias)
    # step4: 打印生成结果
    print(Linear.weight.data)
    print(Linear.bias.data)

# 6.kaiming初始化
def dm06():
    # setp1: 创建线性层,输入5个神经元,输出3个神经元
    Linear = nn.Linear(5, 3)

    # step2: 对权重(w)进行kaiming初始化,产生参数
    # kaiming正态分布初始化
    nn.init.kaiming_normal_(Linear.weight)
    # kaiming均匀分布初始化
    # nn.init.kaiming_uniform_(Linear.weight)

    # step3: 打印生成结果
    print(Linear.weight.data)

# 7.xavier初始化
def dm07():
    # setp1: 创建线性层,输入5个神经元,输出3个神经元
    Linear = nn.Linear(5, 3)

    # step2: 对权重(w)进行xavier初始化,产生参数
    # xavier正态分布初始化
    # nn.init.xavier_normal_(Linear.weight)
    # xavier均匀分布初始化
    nn.init.xavier_uniform_(Linear.weight)

    # step3: 打印生成结果
    print(Linear.weight.data)

# 测试
if __name__ == '__main__':
    # dm01()      # 均匀分布随机初始化
    # dm02()      # 固定初始化
    # dm03()      # 全0初始化
    # dm04()      # 全1初始化
    # dm05()      # 正态分布随机初始化
    # dm06()      # kaiming初始化
    dm07()        # xavier初始化