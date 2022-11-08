# 全连接网络处理多尺度特征
import numpy as np
import torch
import matplotlib.pyplot as plt

# prepare dataset
xy = np.loadtxt('diabetes.csv', delimiter=',', dtype=np.float32)
x_data = torch.from_numpy(xy[:, :-1])  # 第一个‘：’是指读取所有行，第二个‘：’是指从第一列开始，最后一列不要
y_data = torch.from_numpy(xy[:, [-1]])  # [-1] 最后得到的是个矩阵

# design model using class
# 全连接网络
class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.linear1 = torch.nn.Linear(8, 6)  # 输入数据x的特征是8维，x有8个特征
        self.linear2 = torch.nn.Linear(6, 4)
        self.linear3 = torch.nn.Linear(4, 2)
        self.linear4 = torch.nn.Linear(2, 1)
        self.relu = torch.nn.ReLU()  # 将其看作是网络的一层，而不是简单的函数使用

    def forward(self, x):
        x = self.relu(self.linear1(x))
        x = self.relu(self.linear2(x))
        x = self.relu(self.linear3(x))  # y hat
        x = self.relu(self.linear4(x))
        return x


model = Model()
# construct loss and optimizer
# criterion = torch.nn.BCELoss(size_average = True)
criterion = torch.nn.BCELoss(reduction='mean')
optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

epoch_list = []
loss_list = []
# training cycle forward, backward, update
for epoch in range(10000):
    y_pred = model(x_data)
    loss = criterion(y_pred, y_data)
    # 看看损失函数
    print(epoch, loss.item())
    epoch_list.append(epoch)
    loss_list.append(loss.item())
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

# 展示loss和epoch的对应曲线
plt.plot(epoch_list, loss_list)
plt.ylabel('loss')
plt.xlabel('epoch')
plt.show()

# 输出每一个样本的概率 顺便学一下tensor的遍历
y_pred2 = model(x_data)
# print(y_pred2.shape)
dim0,dim1=y_pred2.shape

for i in range(dim0):
    for j in range(dim1):
     print('第',i+1,'个样本的概率为',y_pred2[i][j].item())




# # 查看每一层的参数：
# layer1_weight = model.linear1.weight.data
# layer1_bias = model.linear1.bias.data
# print("layer1_weight", layer1_weight)
# print("layer1_bias", layer1_bias)

# # 查看每一层的参数：
# layer3_weight = model.linear3.weight.data
# layer3_bias = model.linear3.bias.data
# print("layer3_weight", layer3_weight)
# # print("layer3_weight.shape", layer3_weight.shape)
# print("layer3_bias", layer3_bias)
# # print("layer3_bias.shape", layer3_bias.shape)

