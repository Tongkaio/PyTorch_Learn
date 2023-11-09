# 模型的保存
import torch
import torchvision
from torch import nn

vgg16 = torchvision.models.vgg16(pretrained=False)  # 导入未训练的模型（如不存在会自动下载）

# 保存方式1 保存了模型 + 参数(这里是没有预先训练过的参数)
torch.save(vgg16, "vgg16_method1.pth")

# 保存方式2 只保存了参数（字典形式）(官方推荐)，不含有网络结构，占用空间会更小(模型越大效果越显著)
torch.save(vgg16.state_dict(), "vgg16_method2.pth")

