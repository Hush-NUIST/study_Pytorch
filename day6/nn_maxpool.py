# 最大池化

import torch
import torchvision
from torch import nn
from torch.nn import MaxPool2d
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
dataset = torchvision.datasets.CIFAR10("./dataset", train=False, download=False,
                                       transform=torchvision.transforms.ToTensor())

dataloader = DataLoader(dataset, batch_size=1)

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        # ceil_mode决定是否舍弃多余边缘
        self.maxpool1 = MaxPool2d(kernel_size=3, ceil_mode=False)

    def forward(self, input):
        output = self.maxpool1(input)
        return output

cnn = CNN()

writer = SummaryWriter("../logs_maxpool")
step = 0

for data in dataloader:
    imgs, targets = data
    writer.add_images("input", imgs, step)
    output = cnn(imgs)
    writer.add_images("output_maxpool", output, step)
    step = step + 1

writer.close()