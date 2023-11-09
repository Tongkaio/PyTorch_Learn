import torch
from torch import nn


class Tudui(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input):
        output = input + 1
        return output


tudui = Tudui()
x = torch.tensor(1.0)  # 将1.0转化为tensor类型
output = tudui(x)  # Module里的call调用了forward，所以可以往里面传参数
print(output)
