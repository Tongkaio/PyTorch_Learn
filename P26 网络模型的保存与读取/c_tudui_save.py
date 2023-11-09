import torch
from torch import nn
'''
    c_tudui_save.py 和 c_tudui_load.py 演示了一种容易犯的错误：
    当使用「方式1」保存时，保存了模型和参数，当另一个文件要读取pth时，需要在
    那个文件里也定义一下这个模型(下方的class Tudui())，否则会报错
'''


class Tudui(nn.Module):
    def __init__(self):
        super(Tudui, self).__init__()
        self.conv = nn.Conv2d(3, 64, kernel_size=3)

    def forward(self, x):
        x = self.conv1(x)
        return x


tudui = Tudui()
torch.save(tudui, "tudui_method1.pth")