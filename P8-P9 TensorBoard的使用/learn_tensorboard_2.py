from torch.utils.tensorboard import SummaryWriter
import numpy as np
from PIL import Image
# opencv读取的图片类型是numpy型


writer = SummaryWriter("../logs")  # 存储到logs文件夹下
image_path = "../dataset/train/ants_image/0013035.jpgs"
img_PIL = Image.open(image_path)
img_array = np.array(img_PIL)  # 转换成numpy.array类型
# print(type(img_array))  #验证确实是narray类型的
# print(img_array.shape)  #看一下shape的格式（512，768，3）
# 从PIL到numpy，需要在add_image()中指定shape中每一个数字/维的含义，即dataformats = "HWC"
writer.add_image("test", img_array, 1, dataformats="HWC")  # 需要的类型必须是numpy.array torch.tensor string等类型的，所以前面需要进行类型转换

# 画图y = x
for i in range(100):
    writer.add_scalar("y=x", i, i)  # 添加一个标量数据

writer.close()
