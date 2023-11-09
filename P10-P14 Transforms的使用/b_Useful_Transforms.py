from PIL import Image
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
# 常见的Transforms

# 配置 tensorboard
writer = SummaryWriter("../logs")
img = Image.open("../dataset/train/ants_image/0013035.jpg")
print(img)

# ToTensor: 将其他类型转换为 tensor 类型
trans_totensor = transforms.ToTensor()
img_tensor = trans_totensor(img)
# writer.add_image("ToTensor", img_tensor)
print(img_tensor.shape)


#  Normalize: 归一化
print(img_tensor[0][0][0])
trans_norm = transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
img_norm = trans_norm(img_tensor)
print(img_norm[0][0][0])
writer.add_image("Normalize", img_norm)

# Resize
print(img.size)
trans_resize = transforms.Resize((512, 512))
# img PIL -> resize -> PIL
img_resize = trans_resize(img)
# img_resize PIL -> totensor -> tensor
img_resize_tensor = trans_totensor(img_resize)
writer.add_image("Resize", img_resize_tensor)
print(img_resize)

# Compose - resize - 2
trans_resize_2 = transforms.Resize(512)

# PIL -> PIL -> tensor
# 注意compose()列表里的类的顺序不能错，即前一个类的输出类型要和下一个类的输入类型相匹配
# 具体说就是trans_resize_2输出一个PIL类，trans_totensor的输入要求是PIL类，然后把PIL类转化为tensor类
# 转化为tensor类后就可以用tensorboard查看了
# 但是现在resize的输入也可以是tensor类型的了，所以下面的括号里，调换顺序也没错误
trans_compose = transforms.Compose([trans_resize_2, trans_totensor])
img_resize_2 = trans_compose(img)
writer.add_image("Resize2", img_resize_2)


# RandomCrop
trans_random = transforms.RandomCrop(512)
trans_compose_2 = transforms.Compose([trans_random, trans_totensor])
for i in range(10):
    img_crop = trans_compose_2(img)
    writer.add_image("RandomCrop", img_crop, i)

writer.close()
# 在命令行输入 tensorboard --logdir=logs 打开tensorboard
