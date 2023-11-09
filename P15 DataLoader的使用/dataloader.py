import torchvision
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter


# 准备的测试数据集
test_data = torchvision.datasets.CIFAR10("../dataset2", train=False, transform=torchvision.transforms.ToTensor())

'''DataLoader:
    batch_size: 一次载入数量
    shuffle: 两次数据集内容顺序是否一致, True打乱
    num_workers = 0: 用主进程进行加载, 多进程用的, num_works > 0 时在 windows 系统下可能会出错！
    drop_last: 最后剩了几个数据集是否舍去, False保留
'''

test_loader = DataLoader(dataset=test_data, batch_size=64, shuffle=True, num_workers=0, drop_last=False)

# 测试数据集中第一张图片
img, target = test_data[0]
print(img.shape)
print(target)

writer = SummaryWriter("../logs_dataloader")

# shuffle=True两轮不一样, False两轮一样
for epoch in range(2):  # epoch会从0变到1
    step = 0
    for data in test_loader:
        imgs, targets = data  # imgs是64张图片压缩成的[64,_,_,_]的tensor, target是对应的标签
        # print(img.shape)
        # print(targets)
        # exit()
        writer.add_images("Epoch:{}".format(epoch), imgs, step)
        step = step + 1

writer.close()
