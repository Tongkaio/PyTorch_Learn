import torch
from torch import nn
'''
    需要定义下面这个 Tudui 模型，或者用 from c_tudui_save import * 引入
'''


class Tudui(nn.Module):
    def __init__(self):
        super(Tudui, self).__init__()
        self.conv = nn.Conv2d(3, 64, kernel_size=3)

    def forward(self, x):
        x = self.conv1(x)
        return x


# 不需要写：tudui = Tudui()，可以直接导入模型+参数
model = torch.load('tudui_method1.pth')
print(model)
