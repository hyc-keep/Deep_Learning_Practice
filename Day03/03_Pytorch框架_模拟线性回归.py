#导入相关模块
import torch
from torch.utils.data import TensorDataset          #构造数据集对象
from torch.utils.data import DataLoader             #数据加载器
from torch import nn                                #nn模块中有平方损失函数和假设函数
from torch import optim                             #optim模块中有优化器函数
from sklearn.datasets import make_regression        #创建线性回归模型数据集(sklearn机器学习工具箱,包含make_regression 数据生成模块)
import matplotlib.pyplot as plt                     #可视化

plt.rcParams['font.sans-serif'] = ['SimHei']        #用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False          #用来正常显示负号

#1.定义函数,创建线性回归模型数据集
def create_dataset():
    x,y,coef = make_regression(
        n_samples=100,       #样本数量
        n_features=1,        #特征数量
        bias=14.5,           #偏置
        noise=10,            #噪声,噪声越大,样本点越散,噪声越小,样本点越集中
        coef=True,           #是否生成系数
        random_state=3       #随机数种子
    )

    #2.把上述数据转换成张量
    x = torch.tensor(x,dtype=torch.float32)
    y = torch.tensor(y,dtype=torch.float32)
    coef = torch.tensor(coef, dtype=torch.float32)

    #3.返回结果
    return x,y,coef

#2.定义函数,表示模型训练
def train(x,y,coef):
    #1.创建数据集对象,把张量-->数据集对象-->数据加载器
    dataset = TensorDataset(x,y)

    #2.创建数据加载器对象
    #参1:数据集对象  参2:批次大小,一批16条样本数据  参3:是否打乱数据(训练集打乱,测试集不打乱(因为不影响预测))
    dataloader = DataLoader(dataset,batch_size=16,shuffle=True)

    #3.创建初始的模型(线性回归模型)
    #参1:输入特征维度,输入一个x  参2:输出标签维度,输出一个y
    model = nn.Linear(1,1)

    #4.创建损失函数对象
    criterion = nn.MSELoss()    #平方损失函数

    #5.创建优化器对象
    #参1:模型参数,也就是前面配置模型的一个输入x和一个输出y    参2:学习率
    optimizer = optim.SGD(model.parameters(),lr=0.01)

    #6.具体的训练过程
    #6.1 定义变量,分表表示:训练轮数,每轮(平均)损失列表,一轮中训练的总损失值(小数,因为除),一轮中训练的样本数
    epochs,loss_list = 100,[]

    #6.2  开始训练,按轮训练
    for epoch in range(epochs):     #epoch的值从0开始,到99结束      具体每轮训练的动作
        #每轮开始前,重置总损失和样本计数
        total_loss, total_sample = 0.0,0
        # 6.3 每轮是 按批次 训练的,所以从 数据加载器中 获取 批次数据      每轮每批的训练动作
        for train_x,train_y in dataloader:      #7批(16,16,16,16,16,16,4)    每次拿16条数据进行训练
            #6.4模型预测
            y_pred = model(train_x)
            #6.5计算(每批的平均)损失
            loss = criterion(y_pred,train_y.reshape(-1,1))   #计算每批16条样本的平均损失,y要和x矩阵对齐.(-1,1)表示能转多少行就转多少行,1列
            #6.6计算总损失 和 总样本数
            total_loss += loss.item() * train_x.size(0)  # 每一批的平均损失 × 样本数 = 该批总损失
            total_sample += train_x.size(0)       # 累加实际样本数量 train_x.size(0)表示批次x矩阵的行数,也就是实际样本数量
            #6.7 梯度清零 + 反向传播 + 优化器更新梯度参数
            optimizer.zero_grad()   #梯度清零
            loss.backward()         #反向传播,梯度计算
            optimizer.step()        #优化器更新梯度参数

        #6.8 本轮训练结束,把本轮(平均)损失保存到列表中
        loss_list.append(total_loss / total_sample)
        print(f"第{epoch + 1}轮,平均损失:{total_loss / total_sample:.5f}")

    #7.打印最终训练结果
    print(f"{epochs}轮的平均损失列表为:{loss_list}")
    print(f"权重:{model.weight},偏置:{model.bias}")

    #8.绘制损失曲线
    plt.plot(range(1,epochs + 1),loss_list)
    plt.title("损失值曲线变化图")
    plt.xlabel("训练轮数")
    plt.ylabel("平均损失")
    plt.grid()
    plt.show()

    #9.绘制预测值和真实值的关系
    #9.1绘制样本点分布情况
    plt.scatter(x, y, label="样本数据")
    # 9.2使用模型进行预测
    y_pred = model(x)

    #9.3计算真实值
    y_true = coef * x + 14.5
    #9.4绘制预测值和真实值折线图
    plt.plot(x, y_pred.detach(), 'r-', linewidth=2, label="预测值")    #y_pred.detach()表示不跟踪梯度
    plt.plot(x, y_true, 'g--', linewidth=2, label="真实值")
    #9.5图例,网格
    plt.legend()
    plt.grid()
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.title("线性回归拟合效果对比")
    #9.6显示图像
    plt.show()

#3.测试
if __name__ == '__main__':
    #3.1创建数据集
    x,y,coef = create_dataset()
    print(f"x:{x},y:{y},coef:{coef}")

    #3.2模型训练
    train(x,y,coef)
