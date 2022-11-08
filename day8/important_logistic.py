#logistic回归 即多一个sigmod的线性回归 做标签为0or1的二元分类
# logistic回归是一种广义线性回归（generalized linear model），
# 因此与多重线性回归分析有很多相同之处。它们的模型形式基本上相同，都具有 w‘x+b，其
# 中w和b是待求参数，其区别在于他们的因变量不同，多重线性回归直接将w‘x+b作为因变量，
# 即y =w‘x+b，而logistic回归则通过函数L将w‘x+b对应一个隐状态p，p =L(w‘x+b),
# 然后根据p 与1-p的大小决定因变量的值。如果L是logistic函数，就是logistic回归，
# 如果L是多项式函数就是多项式回归。
import torch
from torch.utils.tensorboard import SummaryWriter

# import torch.nn.functional as F

# prepare dataset
x_data = torch.Tensor([[1.0], [2.0], [3.0]])
# 真实样本标签 sigmod 出来是0-1的值 所以样本设置为二元分类
y_data = torch.Tensor([[0], [0], [1]])


# design model using class
class LogisticRegressionModel(torch.nn.Module):
    def __init__(self):
        super(LogisticRegressionModel, self).__init__()
        self.linear = torch.nn.Linear(1, 1)
        self.Sigmoid=torch.nn.Sigmoid()

    def forward(self, x):
        # y_pred = F.sigmoid(self.linear(x))
        # 模型预测值就是多了个激活函数套着线性回归
        # 这里有不同的Sigmoid用法torch.sigmoid&torch.nn.sigmoid  我这个写法当作网络的一层 在tensorboard里能看出差别
        # y_pred = torch.sigmoid(self.linear(x))
        y_pred=self.Sigmoid(self.linear(x))
        print('y_pred',y_pred)
        return y_pred


model = LogisticRegressionModel()

# construct loss and optimizer
# 默认情况下，loss会基于element平均，如果size_average=False的话，loss会被累加。
criterion = torch.nn.BCELoss(size_average=False)
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

# training cycle forward, backward, update
for epoch in range(1000):
    y_pred = model(x_data)
    loss = criterion(y_pred, y_data)
    print(epoch, loss.item())
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

print('w = ', model.linear.weight.item())
print('b = ', model.linear.bias.item())
# 测试集
x_test = torch.Tensor([[4.0]])
y_test = model(x_test)
x_test2 = torch.Tensor([[1.1]])
y_test2 = model(x_test2)
print('y_pred1 = ', y_test.data)
print('y_pred2 = ', y_test2.data)


y_test3 = model(x_data)
print('y_pred3 = ', y_test3.data)

writer = SummaryWriter("../logs_logistic")
writer.add_graph(model, x_test)
