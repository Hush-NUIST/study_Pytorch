# 学习使用Dataset
import torch
import numpy as np
from torch.utils.data import Dataset
from torch.utils.data import DataLoader


# prepare dataset


class DiabetesDataset(Dataset):
    # 必须实现的三个方法 初始化 获取样本 长度
    def __init__(self, filepath):
        # 如何加载文本文件 以逗号为分隔符
        xy = np.loadtxt(filepath, delimiter=',', dtype=np.float32)
        self.len = xy.shape[0]  # shape(多少行，多少列)
        # [start : end : step] 第一个到倒数第一个 最后一个
        self.x_data = torch.from_numpy(xy[:, :-1])
        self.y_data = torch.from_numpy(xy[:, [-1]])

    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    def __len__(self):
        return self.len


dataset = DiabetesDataset('diabetes.csv')
train_loader = DataLoader(dataset=dataset, batch_size=32, shuffle=True, num_workers=2)  # num_workers 多线程


# design model using class

class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.linear1 = torch.nn.Linear(8, 6)
        self.linear2 = torch.nn.Linear(6, 4)
        self.linear3 = torch.nn.Linear(4, 1)
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, x):
        x = self.sigmoid(self.linear1(x))
        x = self.sigmoid(self.linear2(x))
        x = self.sigmoid(self.linear3(x))
        return x


model = Model()

# construct loss and optimizer 指定reduction=‘mean’，求均值，reduction默认为’mean’ =none 结果为tensor =sum  结果为求和
criterion = torch.nn.BCELoss(reduction='sum')
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

# training cycle forward, backward, update
if __name__ == '__main__':
    for epoch in range(100):
        for i, data in enumerate(train_loader, 0):  # train_loader 是先shuffle后mini_batch
            inputs, labels = data
            y_pred = model(inputs)
            loss = criterion(y_pred, labels)
            print('第',epoch+1,'轮  ','第', i,'个batch', loss.item())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()