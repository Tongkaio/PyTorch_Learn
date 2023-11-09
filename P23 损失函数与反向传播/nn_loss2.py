import torchvision
from torch import nn
from torch.nn import Conv2d, MaxPool2d, Flatten, Linear, Sequential
from torch.utils.data import DataLoader


# CIFAR10 网络模型
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
            Linear(1024, 64),
            Linear(64, 10),
        )

    def forward(self, x):
        x = self.model1(x)
        return x


if __name__ == '__main__':
    dataset = torchvision.datasets.CIFAR10("../dataset2", train=False, download=True,
                                           transform=torchvision.transforms.ToTensor())
    dataloader = DataLoader(dataset, batch_size=1, drop_last=True)

    loss = nn.CrossEntropyLoss()
    tudui = Tudui()
    for data in dataloader:
        imgs, targets = data
        outputs = tudui(imgs)
        result_loss = loss(outputs, targets)  # 计算 loss
        result_loss.backward()  # 反向传播
        # print(result_loss)
        # print(outputs)
        # print(targets)
        print('OK')
