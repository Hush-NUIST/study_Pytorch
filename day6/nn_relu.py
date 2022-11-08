# RELU非线性激活函数
import torch
import torchvision
from torch import nn
from torch.nn import ReLU, Sigmoid
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

input = torch.tensor([[1, -0.5],
                      [-1, 3]])

input = torch.reshape(input, (-1, 1, 2, 2))
print(input.shape)

dataset = torchvision.datasets.CIFAR10("./dataset", train=False, download=False,
                                       transform=torchvision.transforms.ToTensor())

dataloader = DataLoader(dataset, batch_size=4)

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        # input=-1 inplace=True input=0   inplace=False input=-1 output=0 是否替换
        self.relu1 = ReLU()
        self.sigmoid1 = Sigmoid()

    def forward(self, input):
        output = self.sigmoid1(input)
        return output

cnn = CNN()

writer = SummaryWriter("../logs_sigmod")
step = 0
for data in dataloader:
    imgs, targets = data
    writer.add_images("input", imgs, global_step=step)
    output = cnn(imgs)
    writer.add_images("output", output, step)
    step += 1

writer.close()


