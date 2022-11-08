import torch
import torch.nn as nn
from matplotlib import pyplot as plt
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
from torchvision import datasets
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torch.optim as optim

# ******************************************************准备数据集********************************************************
# 标准化（Normalization）是神经网络对数据的一种经常性操作。标准化处理指的是：样本减去它的均值，再除以它的标准差，最终样本将呈现均值为 0 方差为 1 的数据分布。
# 神经网络模型偏爱标准化数据，原因是均值为0方差为1的数据在 sigmoid、tanh 经过激活函数后求导得到的导数很大，反之原始数据不仅分布不均（噪声大）而且数值通常都很
# 大（本例中数值范围是 0~255），激活函数后求导得到的导数则接近与 0，这也被称为梯度消失。所以说，数据的标准化有利于加快神经网络的训练。
# 除此之外，还需要保持 train_set、val_set 和 test_set 标准化系数的一致性。标准化系数就是计算要用到的均值和标准差，在本例中是((0.1307,), (0.3081,))，
# 均值是 0.1307，标准差是 0.3081，这些系数都是数据集提供方计算好的数据。不同数据集就有不同的标准化系数，例如([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
# 就是 ImageNet dataset 的标准化系数（RGB三个通道对应三组系数），当需要将 Imagenet 预训练的参数迁移到另一神经网络时，被迁移的神经网络就需要使用 Imagenet的系数，
# 否则预训练不仅无法起到应有的作用甚至还会帮倒忙。
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])  # 归一化,均值和方差
train_dataset = datasets.MNIST(root='./dataset/mnist_train/', train=True, download=True, transform=transform)
train_loader = DataLoader(train_dataset, shuffle=False, batch_size=64)
test_dataset = datasets.MNIST(root='./dataset/mnist_test/', train=False, download=True, transform=transform)
test_loader = DataLoader(test_dataset, shuffle=False, batch_size=64)


# ******************************************************搭建模型*********************************************************
# 残差模块
class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.channels = channels
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)

    def forward(self, x):
        # 第一层wight layer 先卷积再relu（x)
        y = F.relu(self.conv1(x))
        # 第二层weight layer 先卷积后和输入相加再rulu（F(x)+x)
        y = self.conv2(y)
        return F.relu(x + y)


class Simple_ResNet(nn.Module):
    def __init__(self):
        super(Simple_ResNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=5)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=5)  # 88 = 24x3 + 16
        self.rblock1 = ResidualBlock(16)
        self.rblock2 = ResidualBlock(32)
        self.mp = nn.MaxPool2d(2)
        self.fc = nn.Linear(512, 10)

    def forward(self, x):
        in_size = x.size(0)

        x = self.mp(F.relu(self.conv1(x)))
        x = self.rblock1(x)
        x = self.mp(F.relu(self.conv2(x)))
        x = self.rblock2(x)

        x = x.view(in_size, -1)
        x = self.fc(x)
        return x


# **************************************************实例化模型、搭建优化器和损失函数******************************************
model = Simple_ResNet()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
loss = torch.nn.CrossEntropyLoss()
loss = loss.to(device)
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.5)


# *****************************************************设置训练过程*******************************************************
def train(epoch):
    print('训练集图片数量: ', len(train_dataset))
    epoch_list = []
    loss_list = []
    for i in range(epoch):
        epoch_list.append(i)
        train_loss_sum = 0.0
        for data in train_loader:
            inputs, target = data
            inputs,target=inputs.to(device),target.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            train_loss = loss(outputs, target)
            train_loss.backward()
            optimizer.step()
            train_loss_sum += train_loss.item()
        print('第', i + 1, '轮训练的LOSS为', train_loss_sum)
        loss_list.append(train_loss_sum)
        # 展示loss和epoch的对应曲线
    plt.plot(epoch_list, loss_list)
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.show()

# *******************************************************设置测试过程*******************************************************
def test():
    print('训练集图片数量: ', len(test_dataset))
    test_loss = 0
    right = 0
    total = 0
    mark=0
    writer = SummaryWriter("./logs_Resnet")
    with torch.no_grad():
        for data in test_loader:
            images, labels = data
            images, labels=images.to(device),labels.to(device)
            outputs = model(images)
            test_loss += loss(outputs, labels).item()
            value, predicted = torch.max(outputs.data, dim=1)
            total += labels.size(0)
            right += (predicted == labels).sum().item()
            writer.add_images("test", images, mark)
            mark=mark+1
    # tensorboard ––logdir=E:\program\python_project\study_Pytorch\ResNet_imageclassIfIcation\logs_Resnet
    writer.add_graph(model, images)
    acc = right / total
    print('预测正确样本数量', right)
    print('测试loss为', test_loss, '  acc= ', acc)



if __name__ == '__main__':
    print('gpu=', torch.cuda.is_available())
    test()
    train(10)
    test()
    # train(20)
    # test()
    # train(40)
    # test()
