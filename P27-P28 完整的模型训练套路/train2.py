# 完整模型训练套路（二）加入验证环节, 规范打印日志, 输出tensorboard日志
import torchvision
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
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

# 2. 用 Dataloader 来加载数据集
train_dataloader = DataLoader(train_data, batch_size=64)
test_dataloader = DataLoader(test_data, batch_size=64)

# 3. 创建网络模型, 类的定义在 model.py 下
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

# 7. 添加tensorboard
writer = SummaryWriter("../logs_train")

for i in range(epoch):
    print("--------第{}轮训练开始--------".format(i+1))
    # 训练步骤开始
    tudui.train()
    for data in train_dataloader:
        imgs, targets = data  # 获取一个batchsize大小的imgs和targets
        outputs = tudui(imgs)  # imgs经过网络得到输出
        loss = loss_fn(outputs, targets)  # 用输出和targets来计算loss

        # 优化器优化模型
        optimizer.zero_grad()  # 梯度清零
        loss.backward()  # 计算梯度
        optimizer.step()  # 根据梯度优化网络参数
        total_train_step = total_train_step + 1  # 训练次数加1
        if total_train_step % 100 == 0:  # 每训练100次打印1次Loss
            print("训练次数: {}, Loss: {}".format(total_train_step, loss.item())) # .item()会把tensor类型的数字转换为对应的整型或浮点型
            writer.add_scalar("train_loss", loss.item(), total_train_step)

    # 验证步骤开始
    tudui.eval()
    total_test_loss = 0
    total_accuracy = 0
    with torch.no_grad():  # 测试时不需要对模型进行优化，所以这里让节点不求梯度，从而节省内存空间
        for data in test_dataloader:
            imgs, targets = data
            outputs = tudui(imgs)
            loss = loss_fn(outputs, targets)  # 计算一个batchsize大小的数据产生的loss
            total_test_loss = total_test_loss + loss.item()  # 计算loss的总和
            # 统计分类正确的数量 (计算得分最大值所对应标签和targets相同的数量)
            accuracy = (outputs.argmax(1) == targets).sum()  # argmax(1)是计算每行最大值的位置，outputs每行是单张图像在不同类别上的score
            total_accuracy = total_accuracy + accuracy

    # 打印日志，保存tensorboard的logs
    print("整体验证集上的loss: {}".format(total_test_loss))
    print("整体验证集上的正确率: {}".format(total_accuracy/test_data_size))
    writer.add_scalar("test_loss", total_test_loss, total_test_step)
    writer.add_scalar("test_accuracy", total_accuracy/test_data_size, total_test_step)
    total_test_step = total_test_step + 1

    torch.save(tudui, "tudui_{}.pth".format(i))  # 每一轮保存一次模型文件, 方式1
    # torch.save(tudui.state_dict(), "tudui_{}.pth".format(i))  # 方式2
    print("模型已保存")

writer.close()
