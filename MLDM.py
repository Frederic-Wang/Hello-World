import numpy as np
import  matplotlib.pyplot as plt
import torch
import  torch.nn as nn
import torch.nn.functional as Fn


# ====================== 生成数据
sample_nums = 100
mean_value = 2.0

x3 = np.linspace(-2, 7, 12)
target_function = -x3 + 1  # 目标函数为 x + y - 1 = 0

n_data = torch.ones(sample_nums, 1) # 确定生成数据数量

x0 = torch.normal(mean_value * n_data, 1)
y0 = -x0 + torch.ones(sample_nums,1) + torch.randint(1,5,(sample_nums,1)) # 在目标函数的基础上生成数据，方便可视化
t0 = torch.cat((x0, y0), 1) # 在行方向上对两个坐标向量集进行拼接，方便后续代码处理
u0 = torch.zeros(sample_nums) # u0表示数据t0被分类为0

x1 = torch.normal(mean_value * n_data, 1)
y1 = -x1 + torch.ones(sample_nums,1) - torch.randint(1,5,(sample_nums,1))
t1 = torch.cat((x1, y1), 1) # 在行方向上对两个坐标向量集进行拼接，方便后续代码处理
u1 = torch.ones(sample_nums) # u1 表示数据t1被分类为1

train_t = torch.cat((t0, t1), 0)   # 列方向上拼接二维坐标向量，方便后续编程
train_u = torch.cat((u0, u1), 0)   # 列方向上拼接一维标签向量，方便后续编程


# ===================== 搭建感知机
class MyPerceptron(nn.Module):
    def __init__(self):
        super(MyPerceptron, self).__init__()
        self.features = nn.Linear(2, 1)    # 输入为二维坐标向量，输出为1维标签
        self.sigmoid = nn.Sigmoid()        # 激活函数选择sigmoid函数

    def forward(self, x):
        # 前向传递，感知机仅含一层，线性函数+激活函数
        x = self.features(x)
        x = self.sigmoid(x)
        return x


my_perceptron = MyPerceptron()   # 实例化感知机模型
print(my_perceptron)
# ===================== 选择损失函数
loss_fn = nn.MSELoss()     # 损失函数选择平方损失函数

# ===================== 选择优化器
lr = 0.01 # 学习率
optimizer = torch.optim.SGD(my_perceptron.parameters(), lr = lr, momentum = 0.9)
# 使用带有动量（即惯性）的梯度下降法

# ===================== 开始模型训练
plt.ion()
for i in range(100):
    y_pred = my_perceptron(train_t)  # 前向传播

    loss = loss_fn(y_pred.squeeze(), train_u) # 计算损失

    loss.backward()   # 误差反向传播

    optimizer.step() # 更新参数

    optimizer.zero_grad() # 清空梯度

    # 绘图
    plt.cla()

    mask = y_pred.ge(0.5).float().squeeze()  # 以0.5为阈值进行分类
    correct = (mask == train_u).sum()  # 分类正确的样本数
    acc = correct.item() / train_u.size(0) # 计算正确率
    plt.scatter(t0.data.numpy()[:, 0], t0.data.numpy()[:, 1], c = 'r', label = 'class 0')
    plt.scatter(t1.data.numpy()[:, 0], t1.data.numpy()[:, 1], c = 'b', label = 'class 1')

    w0, w1 = my_perceptron.features.weight[0] # 获取所得权重
    w0, w1 = float(w0.item()), float(w1.item()) # 将所的权重转为数字形式，tensor形式不好处理
    b = float(my_perceptron.features.bias[0].item()) # 获取线性方程中另外一个参数b
    plot_x = np.linspace(-3,7,20)
    plot_y = (-w0 * plot_x - b) / w1 # w0 * x + w1 * y + b = 0 ---> y = (-w0 * x - b) / w1
    plt.plot(plot_x, plot_y)
    plt.pause(0.2)
    print(acc)

    if(acc > 0.99):
        break

plt.ioff()
plt.show()


