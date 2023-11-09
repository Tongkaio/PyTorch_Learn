from PIL import Image
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms


# 鼠标放置在 Image 上 alt+enter 可以快速导入要用的库
img = Image.open("../dataset/train/ants_image/0013035.jpg")

# transforms.ToTensor 的使用
tensor_trans = transforms.ToTensor()  # 初始化类
tensor_img = tensor_trans(img)  # PIL → tensor
print(tensor_img)

# 生成日志
writer = SummaryWriter("../logs")
writer.add_image("Tensor_img", tensor_img)  # tensor
writer.close()
