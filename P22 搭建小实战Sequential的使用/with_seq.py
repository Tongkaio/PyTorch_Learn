import torch
from torch import nn
from torch.nn import Conv2d, MaxPool2d, Flatten, Linear, Sequential
from torch.utils.tensorboard import SummaryWriter


# 使用 Sequential 实现 CIFAR10 网络模型
class Tudui(nn.Module):
    def __init__(self):
        super(Tudui, self).__init__()
        self.model1 = Sequential(
            Conv2d(3, 32, 5, padding=2),
            MaxPool2d(2),
            Conv2d(32, 32, 5, padding=2),
            MaxPool2d(2),
            Conv2d(32, 64, 5, padding=2),
            MaxPool2d(2),
            Flatten(),
            Linear(1024, 64),  # 64*4*4
            Linear(64, 10),
        )

    def forward(self, x):
        x = self.model1(x)
        return x


if __name__ == '__main__':
    tudui = Tudui()
    print(tudui)
    input = torch.ones((64, 3, 32, 32))  # batchsize=64
    output = tudui(input)
    print(output.shape)

    writer = SummaryWriter("../logs")
    writer.add_graph(tudui, input)  # 将(模型, 输入)填入，可以可视化整个网络结构，图见readme
    writer.close()
