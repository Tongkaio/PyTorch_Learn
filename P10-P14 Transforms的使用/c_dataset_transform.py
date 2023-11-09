import torchvision
from torch.utils.tensorboard import SummaryWriter

dataset_transform = torchvision.transforms.Compose([  # 把数据集转化为tensor类型
    torchvision.transforms.ToTensor()
])


# download始终开启为True，下载可以用迅雷下载
train_set = torchvision.datasets.CIFAR10(root="../dataset2", train=True, transform=dataset_transform, download=True)
test_set = torchvision.datasets.CIFAR10(root="../dataset2", train=False, transform=dataset_transform, download=True)

# print(test_set[0])
# print(test_set.classes)
#
# img, target = test_set[0]
# print(img)
# print(target)
# print(test_set.classes[target])
# img.show()
# print(test_set[0])
writer = SummaryWriter("../logs_p10")
for i in range(10):  # 从0到9
    img, target = test_set[i]
    writer.add_image("test_set", img, i)

writer.close()
