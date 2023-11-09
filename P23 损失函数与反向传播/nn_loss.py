import torch
from torch.nn import L1Loss, MSELoss, CrossEntropyLoss
from torch import nn

'''
[L1Loss](https://pytorch.org/docs/stable/generated/torch.nn.L1Loss.html#torch.nn.L1Loss)
[MSELoss](https://pytorch.org/docs/stable/generated/torch.nn.MSELoss.html#torch.nn.MSELoss)
'''

# 原始输入
inputs = torch.tensor([1, 2, 3], dtype=torch.float32)
targets = torch.tensor([1, 2, 5], dtype=torch.float32)

# reshape
inputs = torch.reshape(inputs, (1, 1, 1, 3))  # 1batch 1channel 1行 3列
targets = torch.reshape(targets, (1, 1, 1, 3))

# 计算L1Loss
loss = L1Loss(reduction='mean')  # 取平均
result = loss(inputs, targets)  # 2/3, 0.6666666865348816

# 计算MSELoss
loss_mse = MSELoss()  # 默认取平均
result_mse = loss_mse(inputs, targets)  # (2^2)/3 = 4/3, 1.3333333730697632

print("L1Loss: {}".format(result))
print("MSELoss: {}".format(result_mse))

# 计算CrossEntropyLoss
x = torch.tensor([0.1, 0.2, 0.4, 0.3])  # output
y = torch.tensor([2])  # target
x = torch.reshape(x, (1, 4))  # 1batchsize 4类
loss_cross = CrossEntropyLoss()
result_cross = loss_cross(x, y)
print("CrossEntropyLoss: {}".format(result_cross))
