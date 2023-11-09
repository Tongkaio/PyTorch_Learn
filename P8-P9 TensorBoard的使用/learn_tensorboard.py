from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter("../logs")  # 存储到logs文件夹下

# 画图y = x
for i in range(100):
    writer.add_scalar("y=x", i, i)  # 添加一个标量数据

writer.close()

'''在命令行输入以下内容来打开 tensorboard:
    tensorboard --logdir=logs
'''