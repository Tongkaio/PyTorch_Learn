# GPU训练方式（二）
import time
import torchvision
import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter


# 0. 定义训练的设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

# 1. 准备数据集
train_data = torchvision.datasets.CIFAR10("../dataset2", train=True, download=True,
                                       transform=torchvision.transforms.ToTensor())
test_data = torchvision.datasets.CIFAR10("../dataset2", train=False, download=True,
                                       transform=torchvision.transforms.ToTensor())

train_data_size = len(train_data)  # 获取数据集有多少张
test_data_size = len(test_data)
print("训练数据集长度为:{}".format(train_data_size))  # 字符串格式化操作
print("验证数据集长度为:{}".format(test_data_size))

# 2. 利用 Dataloader 来加载数据集
train_dataloader = DataLoader(train_data, batch_size=64)
test_dataloader = DataLoader(test_data, batch_size=64)


# 3. 创建网络模型 CIFAR10
class Tudui(nn.Module):
    def __init__(self):
        super(Tudui, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32,kernel_size=(5,5), stride=(1,1), padding='same'),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(5, 5), stride=(1, 1), padding='same'),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(5, 5), stride=(1, 1), padding='same'),
            nn.MaxPool2d(kernel_size=2),
            nn.Flatten(),
            nn.Linear(64*4*4, 64),
            nn.Linear(64, 10)
        )

    def forward(self,x):
        x = self.model(x)
        return x


tudui = Tudui()
# tudui = tudui.to(device)  # 这里其实不需要返回值，直接写 tudui.to(device) 就可以，但文件img targets需要
tudui.to(device)

# 4. 损失函数
loss_fn = nn.CrossEntropyLoss()
loss_fn = loss_fn.to(device)

# 5. 优化器, SGD(随机梯度下降)
learning_rate = 1e-2  # 设置学习率为0.01, 1e-2是更常见的写法
optimizer = torch.optim.SGD(tudui.parameters(), lr=learning_rate)

# 6. 设置训练网络的一些参数
total_train_step = 0  # 记录训练次数
total_test_step = 0  # 记录测试次数
epoch = 50  # 训练的轮数

# 7. 添加tensorboard
writer = SummaryWriter("../logs_train")
start_time = time.time()
for i in range(epoch):
    print("--------第{}轮训练开始--------".format(i+1))
    # 训练步骤开始
    tudui.train()
    for data in train_dataloader:
        imgs, targets = data
        imgs = imgs.to(device)
        targets = targets.to(device)
        outputs = tudui(imgs)
        loss = loss_fn(outputs, targets)
        # 优化器优化模型
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_train_step = total_train_step + 1  # 训练次数加1
        if total_train_step % 100 == 0:
            end_time = time.time()
            print(end_time - start_time)
            print("训练次数:{}, Loss:{}".format(total_train_step, loss.item()))  # .item() 会输出不带tensor字样的tensor数据，不加也行
            writer.add_scalar("train_loss", loss.item(), total_train_step)

    # 验证步骤开始
    tudui.eval()
    total_test_loss = 0
    total_accuracy = 0
    with torch.no_grad():  # 让节点不求梯度，从而节省内存空间
        for data in test_dataloader:
            imgs, targets = data
            imgs = imgs.to(device)  # 用GPU
            targets = targets.to(device)  # 用GPU
            outputs = tudui(imgs)
            loss = loss_fn(outputs, targets)
            total_test_loss = total_test_loss + loss.item()
            accuracy = (outputs.argmax(1) == targets).sum()
            total_accuracy = total_accuracy + accuracy
    print("整体测试集上的loss:{}".format(total_test_loss))
    print("整体验证集上的正确率:{}".format(total_accuracy/test_data_size))
    writer.add_scalar("test_loss", total_test_loss, total_test_step)
    writer.add_scalar("test_accuracy", total_accuracy/test_data_size, total_test_step)
    total_test_step = total_test_step + 1

    torch.save(tudui, "tudui_{}.pth".format(i))  # 每一轮保存一次模型文件,方式1
    # torch.save(tudui.state_dict(), "tudui_{}.pth".format(i))  # 方式2
    print("模型已保存")

writer.close()
