"""
    自定义PyTorch中的Sampler。
    参考文章：https://zhuanlan.zhihu.com/p/165136131
    运行这个脚本以打印结果。
"""
import random
import torch
from torch.utils.data import Dataset, Sampler, DataLoader


class MyDataset(Dataset):
    """自定义的Dataset类。含100个2*2大小的图像，2个类别，标签值为0和1。"""
    def __init__(self):
        self.img = [torch.ones(2, 2) for i in range(100)]  # img是长度为100的list
        self.num_classes = 2  # 2类
        self.label = torch.tensor([random.randint(0, self.num_classes - 1) for i in range(100)])  # label是长度为100的list，值为0和1

    def __getitem__(self, index):
        return self.img[index], self.label[index]

    def __len__(self):
        return len(self.label)


class MySampler(Sampler):  # 自定义 Sampler 要继承 Sampler 这个类
    """自定义Sampler类。将标签0的索引排在前面，标签1的索引排在后面。"""
    def __init__(self, data):
        self.data = data

    def __iter__(self):  # 自定义 Sampler 通常要重写这个方法
        indices = []
        for n in range(self.data.num_classes):
            index = torch.where(self.data.label == n)[0]  # 如果n=0，则index是data.label中所有0的位置索引
            indices.append(index)
        indices = torch.cat(indices, dim=0)  # 将indices合并成一个长向量
        return iter(indices)  # 返回迭代器

    def __len__(self):
        return len(self.data)


if __name__ == '__main__':
    my_dataset = MyDataset()  # 实例化数据集
    my_sampler = MySampler(my_dataset)  # 实例化sampler
    loader_default_sampler = DataLoader(dataset=my_dataset, batch_size=8)  # 使用Dataloader默认的Sampler(按原始标签顺序打印)
    loader_my_sampler = DataLoader(dataset=my_dataset, batch_size=8, sampler=my_sampler)  # 使用自定义Sampler

    print("【for循环打印Dataloader里的标签(每8个标签(batch_size=8)为1组)】")
    print("使用默认Sampler：")
    for imgs, labels in loader_default_sampler:
        print(labels)

    print("\n使用自定义Sampler：")
    for imgs, labels in loader_my_sampler:
        print(labels)

    '''以下代码用于可视化。
    
    自定义sampler的作用在于对数据集的索引按某种方式进行排序。
    例如，MySampler就按照标签0在前，标签1在后的顺序对“索引”进行了排序，
    从而在从DataLoader读取数据时，总是先取出标签为0的数据，然后才取出标签为1的数据。

    '''
    print("\n【可视化】")
    indices = []
    indices.append(torch.where(my_dataset.label == 0)[0])  # 标签值为0的索引
    indices.append(torch.where(my_dataset.label == 1)[0])  # 标签值为1的索引
    indices = torch.cat(indices, dim=0)  # 拼接为长向量
    print(f"原始标签排列顺序：\n"
          f"{my_dataset.label}\n"
          f"标签值为0的索引：\n"
          f"{torch.where(my_dataset.label == 0)[0]}\n"
          f"标签值为1的索引：\n"
          f"{torch.where(my_dataset.label == 1)[0]}\n"
          f"拼接后的索引：\n"
          f"{indices}\n"
          f"使用sampler后的标签排列顺序：\n"
          f"{my_dataset.label[indices]}")
