from torch.utils.data import Dataset
from PIL import Image
import os
# [视频教程1](https://www.bilibili.com/video/BV1hE411t7RN?p=6&vd_source=635cd9b21458e484a3dbdb77768b6ff7)
# [视频教程2](https://www.bilibili.com/video/BV1hE411t7RN?p=7&vd_source=635cd9b21458e484a3dbdb77768b6ff7)


class MyData(Dataset):
    def __init__(self, root_dir, label_dir):  # self指定了这个类里面的“全局变量”，这个类里可以用
        self.root_dir = root_dir  # 根目录
        self.label_dir = label_dir  # 标签目录
        self.path = os.path.join(self.root_dir, self.label_dir)  # 拼接路径
        self.img_path = os.listdir(self.path)  # 获取所有图片名称列表

    def __getitem__(self, idx):  # 读取每一张图片
        img_name = self.img_path[idx]
        img_item_path = os.path.join(self.root_dir, self.label_dir, img_name)
        img = Image.open(img_item_path)  # 打开图像
        label = self.label_dir
        return img, label

    def __len__(self):  # 返回数据有多长
        return len(self.img_path)


if __name__ == '__main__':
    # 蚂蚁蜜蜂分类数据集[下载链接](https://download.pytorch.org/tutorial/hymenoptera_data.zip)
    root_dir = "../dataset/train"
    ants_label_dir = "ants_image"
    bees_label_dir = "bees_image"
    ants_dataset = MyData(root_dir, ants_label_dir)
    bees_dataset = MyData(root_dir, bees_label_dir)

    train_dataset = ants_dataset + bees_dataset
    img, label = ants_dataset[0]  # 通过 [0] 索引，触发并自动调用 __getitem__ 魔法方法
    img.show()
