import torch
import torchvision
from torch import nn
from torch.nn import Conv2d
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter


class Tudui(nn.Module):
    def __init__(self):
        super(Tudui, self).__init__()  # 继承初始化方法 python3可以直接简单这么写  super().__init__()
        self.conv1 = Conv2d(in_channels=3, out_channels=6, kernel_size=(3, 3), stride=(1, 1), padding=0)

    def forward(self, x):
        x = self.conv1(x)
        return x


if __name__ == '__main__':
    # 测试数据集
    dataset = torchvision.datasets.CIFAR10("../dataset2", train=False, transform=torchvision.transforms.ToTensor(),
                                           download=True)
    dataloader = DataLoader(dataset, batch_size=64)  # 一次载入64张图片

    tudui = Tudui()

    writer = SummaryWriter("../logs")
    # 遍历 dataloader 的数据，送入 tudui 进行卷积
    step = 0
    for data in dataloader:
        imgs, targets = data
        output = tudui(imgs)
        print(imgs.shape)  # 看看卷积前的尺寸torch.Size([64, 3, 32, 32])  batch_size, channel, 长, 宽
        print(output.shape)  # 看看卷积后的尺寸torch.Size([64, 6, 30, 30]) 6通道不能显示
        writer.add_images("input", imgs, step)

        output = torch.reshape(output, (-1, 3, 30, 30))  # 为了可以显示，把通道数降为三，多余的层用来扩充batchsize
        writer.add_images("output", output, step)
        step = step + 1

    writer.close()
