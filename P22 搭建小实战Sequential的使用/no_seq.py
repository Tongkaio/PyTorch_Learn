import torch
from torch import nn
from torch.nn import Conv2d, MaxPool2d, Flatten, Linear


# 不使用 Sequential 实现 CIFAR10 模型
class Tudui(nn.Module):
    def __init__(self):
        super(Tudui, self).__init__()
        self.conv1 = Conv2d(3, 32, 5, padding='same')  # 这里padding为2 如果动手算一下的话
        self.maxpool1 = MaxPool2d(2)
        self.conv2 = Conv2d(32, 32, 5, padding='same')
        self.maxpool2 = MaxPool2d(2)
        self.conv3 = Conv2d(32, 64, 5, padding='same')
        self.maxpool3 = MaxPool2d(2)
        self.flatten = Flatten()
        self.linear1 = Linear(1024, 64)  # 64*4*4
        self.linear2 = Linear(64, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = self.maxpool1(x)
        x = self.conv2(x)
        x = self.maxpool2(x)
        x = self.conv3(x)
        x = self.maxpool3(x)
        x = self.flatten(x)
        x = self.linear1(x)
        x = self.linear2(x)
        return x


if __name__ == '__main__':
    tudui = Tudui()
    print(tudui)
    input = torch.ones((64, 3, 32, 32))  # batchsize=64
    output = tudui(input)
    print(output.shape)
