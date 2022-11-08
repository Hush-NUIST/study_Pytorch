# 学习划分数据集 sklearn库
import torch
import numpy as np
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split


raw_data = np.loadtxt('diabetes.csv', delimiter=',', dtype=np.float32)
X = raw_data[:, :-1]
y = raw_data[:, [-1]]

# 划分数据集
Xtrain, Xtest, Ytrain, Ytest = train_test_split(X, y, test_size=0.3)
# totensor transforms是图片totensor
Xtest = torch.from_numpy(Xtest)
Ytest = torch.from_numpy(Ytest)


# 将训练数据集进行批量处理
# prepare dataset
class DiabetesDataset(Dataset):
    def __init__(self, data, label):
        self.len = data.shape[0]  # shape(多少行，多少列)
        self.x_data = torch.from_numpy(data)
        self.y_data = torch.from_numpy(label)

    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    def __len__(self):
        return self.len


train_dataset = DiabetesDataset(Xtrain, Ytrain)

train_loader = DataLoader(dataset=train_dataset, batch_size=32, shuffle=True, num_workers=10)  # num_workers 多线程


# design model using class


class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.linear1 = torch.nn.Linear(8, 6)
        self.linear2 = torch.nn.Linear(6, 4)
        self.linear3 = torch.nn.Linear(4, 2)
        self.linear4 = torch.nn.Linear(2, 1)
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, x):
        x = self.sigmoid(self.linear1(x))
        x = self.sigmoid(self.linear2(x))
        x = self.sigmoid(self.linear3(x))
        x = self.sigmoid(self.linear4(x))
        return x


model = Model()

# construct loss and optimizer
criterion = torch.nn.BCELoss(reduction='mean')
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)


# training cycle forward, backward, update
def train(epoch):
    train_loss = 0.0
    count = 0
    print(train_loader)
    for i, data in enumerate(train_loader, 0):
        print(data)
        inputs, labels = data
        y_pred = model(inputs)
        loss = criterion(y_pred, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        count = i
    # print('第', epoch + 1, '轮  ', '第', i, '个batch',train_loss)
    # if epoch % 2000 == 1999:
    #     print("train loss:", train_loss / count, end=',')


def test():
    # 在使用pytorch时，并不是所有的操作都需要进行计算图的生成（计算过程的构建，以便梯度反向传播等操作）。而对于
    # tensor的计算操作，默认是要进行计算图的构建的，在这种情况下，可以使用 with torch.no_grad():，强制之后
    # 的内容不进行计算图构建。
    with torch.no_grad():
        y_pred = model(Xtest)
        y_pred_label = torch.where(y_pred >= 0.5, torch.tensor([1.0]), torch.tensor([0.0]))
        acc = torch.eq(y_pred_label, Ytest).sum().item() / Ytest.size(0)
        print("test acc:", acc)


if __name__ == '__main__':
    for epoch in range(15):
        train(epoch)
        if epoch % 15 == 14:
            test()
