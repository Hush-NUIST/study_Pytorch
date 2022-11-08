# 全连接层
import torch
import torchvision
from torch import nn
from torch.nn import Linear
from torch.utils.data import DataLoader

dataset = torchvision.datasets.CIFAR10("./dataset", train=False, transform=torchvision.transforms.ToTensor(),
                                       download=False)

dataloader = DataLoader(dataset, batch_size=64,drop_last=True)

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.linear1 = Linear(196608, 10)

    def forward(self, input):
        output = self.linear1(input)
        return output

cnn = CNN()

for data in dataloader:
    imgs, targets = data
    print(imgs.shape)
    # =torch.reshape(img,(1,1,1,-1))
    output = torch.flatten(imgs)
    print(output.shape)
    output = cnn(output)
    print(output.shape)