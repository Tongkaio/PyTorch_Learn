import torch
import torch.nn.functional as F

'''
    使用 nn.functional.conv2d 解释卷积的原理，实际更常用 nn.Conv2d
'''

input = torch.tensor([[1, 2, 0, 3, 1],
                      [0, 1, 2, 3, 1],
                      [1, 2, 1, 0, 0],
                      [5, 2, 3, 1, 1],
                      [2, 1, 0, 1, 1]])

kernel = torch.tensor([[1, 2, 1],
                       [0, 1, 0],
                       [2, 1, 0]])

input = torch.reshape(input, (1, 1, 5, 5))  # batchsize为1（每次输入的图片数量），通道数为1，数据维度5*5
kernel = torch.reshape(kernel, (1, 1, 3, 3))
# print(input.shape)
# print(kernel.shape)

# [conv2d](https://pytorch.org/docs/stable/generated/torch.nn.functional.conv2d.html#torch-nn-functional-conv2d)
output = F.conv2d(input, kernel, stride=1)
print(output)

output2 = F.conv2d(input, kernel, stride=2)
print(output2)

# padding=1 在输入图像四周填充1个像素
output3 = F.conv2d(input, kernel, stride=1, padding=1)
print(output3)
