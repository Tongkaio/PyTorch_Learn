# 模型的读取
import torch
import torchvision
from torch import nn

# 读取方式1 → 对应于保存方式1
model = torch.load("vgg16_method1.pth")  # 读取了模型+参数
print('方式1----------------------------------------------------------------------------')
print(model)

# 读取方式2 → 对应于保存方式2
# model = torch.load("vgg16_method2.pth")  # 仅获得字典类型的参数
vgg16 = torchvision.models.vgg16(pretrained=False)  # 获得模型
vgg16.load_state_dict(torch.load("vgg16_method2.pth"))  # 读取参数并导入到模型中
print('方式2----------------------------------------------------------------------------')
print(vgg16)



