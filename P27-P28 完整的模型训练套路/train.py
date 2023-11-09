# 完整模型训练套路（一）
import torchvision
from torch import nn
from torch.utils.data import DataLoader
from model import *  # 这里放着模型

# 1. 准备数据集
train_data = torchvision.datasets.CIFAR10("../dataset2", train=True, download=True,
                                       transform=torchvision.transforms.ToTensor())
test_data = torchvision.datasets.CIFAR10("../dataset2", train=False, download=True,
                                       transform=torchvision.transforms.ToTensor())

train_data_size = len(train_data)  # 获取数据集有多少张, 数据集通常会定义__len__魔法方法
test_data_size = len(test_data)
print("训练数据集长度为: {}".format(train_data_size))  # 训练数据集长度为: 50000
print("测试数据集长度为: {}".format(test_data_size))  # 测试数据集长度为: 10000

# 2. 用 Dataloader 加载数据集
train_dataloader = DataLoader(train_data, batch_size=64)
test_dataloader = DataLoader(test_data, batch_size=64)

# 3. 创建模型实例, 类的定义在 model.py 下
tudui = Tudui()

# 4. 损失函数
loss_fn = nn.CrossEntropyLoss()

# 5. 优化器, SGD(随机梯度下降)
learning_rate = 1e-2  # 设置学习率为0.01, 1e-2是更常见的写法
optimizer = torch.optim.SGD(tudui.parameters(), lr=learning_rate)

# 6. 设置训练网络的一些参数
total_train_step = 0  # 记录训练次数
total_test_step = 0  # 记录测试次数
epoch = 10  # 训练的轮数

# 7. 开始训练
for i in range(epoch):
    print("--------第{}轮训练开始--------".format(i+1))
    # 训练步骤开始
    for data in train_dataloader:  # 从dataloader中取数据
        imgs, targets = data  # 获取一个batchsize大小的imgs和targets
        outputs = tudui(imgs)  # imgs经过网络得到输出
        loss = loss_fn(outputs, targets)  # 用输出和targets来计算loss

        # 优化器优化模型
        optimizer.zero_grad()  # 梯度清零
        loss.backward()  # 计算梯度
        optimizer.step()  # 根据梯度优化网络参数

        # 打印日志
        total_train_step = total_train_step + 1  # 训练次数+1
        print("训练次数: {}, Loss: {}".format(total_train_step, loss.item()))  # .item() 会把tensor类型的数字转换为对应的整型或浮点型
