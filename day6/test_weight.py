# 搭建CIFAR10的model
import torch
from torch import nn
from torch.nn import Conv2d, MaxPool2d, Flatten, Linear, Sequential
from torch.utils.tensorboard import SummaryWriter


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.conv=nn.Conv2d(1,1,1)

    def forward(self, x):
        x = self.conv(x)
        print(self.conv.weight)
        print(self.conv.bias)
        return x

test = Model()
input = torch.ones((1,1,1,1))
print(f'input{input}')
output = test(input)
print(f'output{output}')


