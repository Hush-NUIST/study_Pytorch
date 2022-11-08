# 自己实现Xception Imagnet太大了 在验证集上训练 在验证集上测试了
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo
import torchvision
from PIL import Image
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from sklearn.model_selection import train_test_split
from torchvision import transforms

# ******************************************************准备数据集********************************************************
# xception的输入是299 如果不是299需要resize
transform = transforms.Compose([transforms.Resize([299, 299]), transforms.ToTensor()])
# ImageFolder torchvision.datasets的子类，用于加载数据。 要求默认数据集已经自觉按照要分配的类型分成了不同的文件夹，一种类型的文件夹下面只存放一种类型的图片
dataset = torchvision.datasets.ImageFolder('G:/PC/Imagenet/ILSVRC-2012/val', transform, )
# 把标签对应成数值
dataset.classes = dataset.class_to_idx
# ZIP函数将二维数组拆成两列 第一列是图片 第二列是标签
imgs, labels = zip(*dataset.imgs)
# 划分数据集 如果数据集没有分train test 或者自己的数据集就可以用这个来划分 imagenet太大了 而且没找到测试集标签 所以我这里也分一下
Xtrain2, Xtest2, Ytrain2, Ytest2 = train_test_split(imgs, labels, test_size=0.1, shuffle=False)
# 还是太大了 再分
Xtrain, Xtest, Ytrain, Ytest = train_test_split(Xtest2, Ytest2, test_size=0.05, shuffle=False)


# 创建自己的数据集 重写torch的Dataset
class GetLoader(torch.utils.data.Dataset):
    # 初始化函数，得到数据
    def __init__(self, data_root, data_label, transform=None):
        self.data = data_root
        self.label = data_label
        self.transform = transform

    # index是根据batchsize划分数据后得到的索引，最后将data和对应的labels进行一起返回
    def __getitem__(self, index):
        data = self.data[index]
        labels = self.label[index]
        # convert('RGB') 是因为数据集中还有单通道的黑白图片
        data = Image.open(data).convert('RGB')
        if self.transform:
            data = transform(data)
        return data, labels

    # 该函数返回数据大小长度，目的是DataLoader方便划分，如果不知道大小，DataLoader会一脸懵逼
    def __len__(self):
        return len(self.data)


# 通过GetLoader将数据进行加载，返回Dataset对象，包含data和labels 这边其实有问题 不能用train_test_split来分 因为我划分数据库的原因 这样训练集的标签为900-990 测试集为990-1000 图像类别根本没有相交 所以这次测试我决定都在训练集上训练并测试
dataset_train = GetLoader(Xtest, Ytest, transform=transform)
dataset_test = GetLoader(Xtest, Ytest, transform=transform)
# 训练集
# dataloader_train = DataLoader(dataset=dataset_train, batch_size=1, shuffle=True, drop_last=True, num_workers=0)
dataloader_train = DataLoader(dataset=dataset_test, batch_size=8, shuffle=True, drop_last=True, num_workers=0)
# 测试集
dataloader_test = DataLoader(dataset=dataset_test, batch_size=8, shuffle=False, drop_last=True, num_workers=0)


# ******************************************************搭建模型*********************************************************
# xception的核心 深度可分离卷积
class SeparableConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, padding=0, dilation=1, bias=False):
        super(SeparableConv2d, self).__init__()
        # 对每一个通道切片卷积
        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size, stride, padding, dilation, groups=in_channels,
                               bias=bias)
        # 逐通道卷积 pointwise
        self.pointwise = nn.Conv2d(in_channels, out_channels, 1, 1, 0, 1, 1, bias=bias)

    def forward(self, x):
        x = self.conv1(x)
        x = self.pointwise(x)
        return x


# 决定是否为残差块
class Block(nn.Module):
    def __init__(self, in_filters, out_filters, reps, strides=1, start_with_relu=True, grow_first=True):
        super(Block, self).__init__()

        if out_filters != in_filters or strides != 1:
            # 是残差块 残差连接
            self.skip = nn.Conv2d(in_filters, out_filters, 1, stride=strides, bias=False)
            self.skipbn = nn.BatchNorm2d(out_filters)
        else:
            self.skip = None

        self.relu = nn.ReLU(inplace=True)
        rep = []

        filters = in_filters
        if grow_first:
            rep.append(self.relu)
            rep.append(SeparableConv2d(in_filters, out_filters, 3, stride=1, padding=1, bias=False))
            rep.append(nn.BatchNorm2d(out_filters))
            filters = out_filters

        for i in range(reps - 1):
            rep.append(self.relu)
            rep.append(SeparableConv2d(filters, filters, 3, stride=1, padding=1, bias=False))
            rep.append(nn.BatchNorm2d(filters))

        if not grow_first:
            rep.append(self.relu)
            rep.append(SeparableConv2d(in_filters, out_filters, 3, stride=1, padding=1, bias=False))
            rep.append(nn.BatchNorm2d(out_filters))

        if not start_with_relu:
            rep = rep[1:]
        else:
            rep[0] = nn.ReLU(inplace=False)

        if strides != 1:
            rep.append(nn.MaxPool2d(3, strides, 1))
        self.rep = nn.Sequential(*rep)

    def forward(self, inp):
        x = self.rep(inp)

        if self.skip is not None:
            skip = self.skip(inp)
            skip = self.skipbn(skip)
        else:
            skip = inp

        x += skip
        return x


# 模型
class Xception(nn.Module):
    """
        Xception optimized for the ImageNet dataset, as specified in
        https://arxiv.org/pdf/1610.02357.pdf
        """

    def __init__(self, num_classes):
        """ Constructor
            Args:
                num_classes: 总分类数
            """
        super(Xception, self).__init__()
        self.num_classes = num_classes
        ################################## 定义 Entry flow ###############################################################
        # 输入229*229*3
        self.conv1 = nn.Conv2d(3, 32, 3, 2, 0, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        # 这里不用nn.relu是因为不想将其当作网络的一层 不然tensorboard出来的网络图是乱的
        # self.relu = torch.relu()
        self.conv2 = nn.Conv2d(32, 64, 3, bias=False)
        self.bn2 = nn.BatchNorm2d(64)

        # Block中的参数顺序：in_filters,out_filters,reps,stride,start_with_relu,grow_first
        self.block1 = Block(64, 128, 2, 2, start_with_relu=False, grow_first=True)
        self.block2 = Block(128, 256, 2, 2, start_with_relu=True, grow_first=True)
        self.block3 = Block(256, 728, 2, 2, start_with_relu=True, grow_first=True)

        ################################### 定义 Middle flow ########################################
        # repeated 8 times
        self.block4 = Block(728, 728, 3, 1, start_with_relu=True, grow_first=True)
        self.block5 = Block(728, 728, 3, 1, start_with_relu=True, grow_first=True)
        self.block6 = Block(728, 728, 3, 1, start_with_relu=True, grow_first=True)
        self.block7 = Block(728, 728, 3, 1, start_with_relu=True, grow_first=True)

        self.block8 = Block(728, 728, 3, 1, start_with_relu=True, grow_first=True)
        self.block9 = Block(728, 728, 3, 1, start_with_relu=True, grow_first=True)
        self.block10 = Block(728, 728, 3, 1, start_with_relu=True, grow_first=True)
        self.block11 = Block(728, 728, 3, 1, start_with_relu=True, grow_first=True)

        #################################### 定义 Exit flow ###############################################################
        self.block12 = Block(728, 1024, 2, 2, start_with_relu=True, grow_first=False)

        self.conv3 = SeparableConv2d(1024, 1536, 3, 1, 1)
        self.bn3 = nn.BatchNorm2d(1536)

        # do relu here
        self.conv4 = SeparableConv2d(1536, 2048, 3, 1, 1)
        self.bn4 = nn.BatchNorm2d(2048)

        self.fc = nn.Linear(2048, num_classes)

        # ------- init weights --------
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def features(self, input):
        ################################## 定义 Entry flow ###############################################################
        x = self.conv1(input)
        x = self.bn1(x)
        x = torch.relu(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = torch.relu(x)

        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)

        ################################### 定义 Middle flow ############################################################
        x = self.block4(x)
        x = self.block5(x)
        x = self.block6(x)
        x = self.block7(x)
        x = self.block8(x)
        x = self.block9(x)
        x = self.block10(x)
        x = self.block11(x)

        #################################### 定义 Exit flow ###############################################################
        x = self.block12(x)

        x = self.conv3(x)
        x = self.bn3(x)
        x = torch.relu(x)

        x = self.conv4(x)
        x = self.bn4(x)
        return x

    def logits(self, features):
        x = torch.relu(features)
        x = F.adaptive_avg_pool2d(x, (1, 1))
        # 这句话的出现就是为了将前面操作输出的多维度的tensor展平成一维，然后
        # 输入分类器，-1是自适应分配，指在不知道函数有多少列的情况下，根据原
        # tensor数据自动分配列数
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x

    def forward(self, input):
        x = self.features(input)
        x = self.logits(x)

        return x


# **************************************************实例化模型、搭建优化器和损失函数******************************************
xception = Xception(1000)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
xception = xception.to(device)
# 二元交叉熵
loss = nn.CrossEntropyLoss()
loss = loss.to(device)
# 优化器
optimizer = torch.optim.SGD(xception.parameters(), lr=0.01, momentum=0.5)


# *****************************************************设置训练过程*******************************************************
def train(epoch):
    print('训练集图片数量: ', len(dataloader_train.dataset))
    epoch_list = []
    loss_list = []
    for i in range(epoch):
        epoch_list.append(i)
        train_loss_sum = 0.0
        for data in dataloader_train:
            imgs, labels = data
            imgs, labels = imgs.to(device), labels.to(device)
            outputs = xception(imgs)
            train_loss = loss(outputs, labels)
            optimizer.zero_grad()
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
    print('测试集图片数量: ', len(dataloader_test.dataset))
    test_loss = 0
    right = 0
    mark=0
    # 在使用pytorch时，并不是所有的操作都需要进行计算图的生成，只是想要网络结果的话就不需要后向传播 ，如果你想通过网络输出的结果去进一步优化网络的话就需要后向传播了。
    # 不使用with torch.no_grad(): 此时有grad_fn = 属性，表示，计算的结果在一计算图当中，可以进行梯度反传等操作。
    # 使用with torch.no_grad(): 表明当前计算不需要反向传播，使用之后，强制后边的内容不进行计算图的构建
    writer = SummaryWriter("./logs_Xception")
    with torch.no_grad():
        for data in dataloader_test:
            imgs, labels = data
            imgs, labels = imgs.to(device), labels.to(device)
            outputs = xception(imgs)
            value, predicted = torch.max(outputs.data, dim=1)
            right += (predicted == labels).sum().item()
            test_loss += loss(outputs, labels).item()
            writer.add_images("test", imgs, mark)
            mark = mark + 1
    writer.add_graph(xception, imgs)
    writer.close()
    acc = right / len(dataloader_test.dataset)
    print('预测正确样本数量', right)
    print('测试loss为', test_loss, '  acc= ', acc)

    # 模型可视化 tensorboard ––logdir=E:\program\python_project\study_Pytorch\Xception_imageclassification\logs_Xception


if __name__ == '__main__':
    print('gpu=', torch.cuda.is_available())
    test()
    train(10)
    test()
    # train(20)
    # test()
    # train(40)
    # test()
