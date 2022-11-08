# 卷积神经网络图片分类
import torch
import torchvision
from torch import nn
from torch.nn import Sequential, Conv2d, MaxPool2d, Flatten, Linear, ReLU, Sigmoid
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter

dataset = torchvision.datasets.CIFAR10("./dataset", train=False, transform=torchvision.transforms.ToTensor(),
                                       download=False)

# 划分数据集
data_x, data_y = dataset.data, dataset.targets
Xtrain, Xtest, Ytrain, Ytest = train_test_split(data_x, data_y, test_size=0.1)
dataloader = DataLoader(dataset=dataset, batch_size=1, shuffle=False)

# model
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.model1 = Sequential(
            Conv2d(3, 32, 5, padding=2),
            MaxPool2d(2),
            Conv2d(32, 32, 5, padding=2),
            MaxPool2d(2),
            Conv2d(32, 64, 5, padding=2),
            MaxPool2d(2),
            Flatten(),
            Linear(1024, 64),
            Linear(64, 10),
            ReLU(),
        )

    def forward(self, x):
        x = self.model1(x)
        return x


model = CNN()
# construct loss and optimizer
#  nn.CrossEntropyLoss已经包含了softmax层 他的输入有标签 按照标签分类 通过全连接层softmax计算概率 再去label建立损失函数
loss = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

# 用测试集数据
dataset.data=Xtest
dataset.targets=Ytest
# test before train
test_loss=0
for data in dataloader:
    imgs, labels = data
    outputs = model(imgs)
    test_loss += loss(outputs, labels).item()
print('训练前测试loss', test_loss)


# 想用dataset实现数据库 失败了 用训练集数据进行训练
dataset.data=Xtrain
dataset.targets=Ytrain
# train
epoch_list = []
loss_list = []
for i in range(2):
    epoch_list.append(i)
    train_loss_sum = 0.0
    for data in dataloader:
        imgs, labels = data
        outputs = model(imgs)
        train_loss = loss(outputs, labels)
        optimizer.zero_grad()
        train_loss.backward()
        optimizer.step()
        train_loss_sum += train_loss.item()
    print('第', i + 1, '轮的LOSS为', train_loss_sum)
    loss_list.append(train_loss_sum)
# 展示loss和epoch的对应曲线
plt.plot(epoch_list, loss_list)
plt.ylabel('loss')
plt.xlabel('epoch')
plt.show()

# 用测试集数据
dataset.data=Xtest
dataset.targets=Ytest
# test after train
test_loss=0
for data in dataloader:
    imgs, labels = data
    outputs = model(imgs)
    test_loss += loss(outputs, labels).item()
print('训练后测试loss', test_loss)

writer = SummaryWriter("../logs_image_classfication")
writer.add_graph(model,imgs)

