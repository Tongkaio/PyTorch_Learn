# 验证模型
import torch
import torchvision.transforms
from PIL import Image
from torch import nn

image_path = "img/img_4.png"  # 一张飞机照片
image = Image.open(image_path)
image = image.convert('RGB')  # 因为PNG文件有四通道，RGB和一个透明度通道，这和要求的”三通道“不符合
print(image)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 网络模型要求输入的图像尺寸是 32*32, 这里 resize 一下
transform = torchvision.transforms.Compose([torchvision.transforms.Resize((32, 32)),
                                            torchvision.transforms.ToTensor()])
image = transform(image)
print(image.shape)


# CIFAR10
# 模型是保存了模型+参数的，所以上面需要把模型定义好
class Tudui(nn.Module):
    def __init__(self):
        super(Tudui, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=(5, 5), stride=(1,1), padding='same'),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(5, 5), stride=(1, 1), padding='same'),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(5, 5), stride=(1, 1), padding='same'),
            nn.MaxPool2d(kernel_size=2),
            nn.Flatten(),
            nn.Linear(64*4*4, 64),
            nn.Linear(64, 10)
        )

    def forward(self,x):
        x = self.model(x)
        return x


# 如果是在gpu上训练的模型，要在cpu上测试，需要加上map_location
model = torch.load("tudui_49.pth", map_location=torch.device('cpu'))
print(model)
image = torch.reshape(image, (1, 3, 32, 32))
# image = image.to(device)

model.eval()
with torch.no_grad():  # 节约内存
    output = model(image)

class_idx = [
    'airplane',
    'automobile',
    'bird',
    'cat',
    'deer',
    'dog',
    'frog',
    'horse',
    'ship',
    'truck'
]

print(f"output = {output}")  # 输出各类得分
print(f"class is '{class_idx[output.argmax(1)]}', target ID = {output.argmax(1).item()}")
