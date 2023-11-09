import torchvision
from torch import nn

# 1. 加载官方提供的网络模型
# 应该会默认下载到 C 盘
vgg16_false = torchvision.models.vgg16(pretrained=False)  # 使用默认参数的模型
vgg16_true = torchvision.models.vgg16(pretrained=True)  # 使用预训练好的模型
print(vgg16_true)
print("--------------------------------------------------------------------------------------------")
dataset = torchvision.datasets.CIFAR10("../dataset2", train=False, download=True,
                                       transform=torchvision.transforms.ToTensor())

'''
    vgg16最后一层，是classifier的第7层(索引为6)，是Linear(in_features=4096, out_features=1000, bias=True)，
    下面使用两种方式修改网络模型
'''

# 2. 使用 add_moduel 修改网络模型
vgg16_true.classifier.add_module('add_linear', nn.Linear(1000, 10))  # 给classifier网络后加一层全连接，使用add_modeule把输出改为10类(CIFAR10是10类)
print(vgg16_true)
print("--------------------------------------------------------------------------------------------")

# 3. 使用索引号修改网络模型
vgg16_false.classifier[6] = nn.Linear(4096, 10)  # 把classifier的第7层全连接的out_feature改为10
print(vgg16_false)
