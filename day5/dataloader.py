#学习dataloader
import torchvision

# 准备的测试数据集
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

test_data = torchvision.datasets.CIFAR10("./dataset", train=False, transform=torchvision.transforms.ToTensor())
# batch_size 一次从数据集中取多少 shuffle打乱再取 drop_last=True最后一次取的数据不够直接舍去 num_workers是否用多线程读取 建立几个进程
test_loader = DataLoader(dataset=test_data, batch_size=4, shuffle=True, num_workers=0, drop_last=True)

# 测试数据集中第一张图片及target target即label
img, target = test_data[0]
print(img.shape)
print(target)

#使用dataloader
writer = SummaryWriter("dataloader")
for epoch in range(2):
    step = 0
    for data in test_loader:
        imgs, targets = data
        print(imgs.shape)
        print(targets)
        writer.add_images("Epoch: {}".format(epoch), imgs, step)
        step = step + 1

writer.close()


