# 搭建网络
import torch
from torch import nn


class nn(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input):
        output = input + 1
        return output


nn = nn()
x = torch.tensor(1.0)
output = nn(x)
print(output)