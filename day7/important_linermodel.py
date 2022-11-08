#CNN那个看不太懂 用最简单的线性回归模型训练熟悉下nn.moudle
# 复习的时候可以看图片 就是 通过前向过程求出LOSS 再反向求梯度 反向传播更新W权重（和b偏置）
import torch

# prepare dataset
# 数据集 x数据 y标签
x_data = torch.tensor([[1.0], [2.0], [3.0]])
y_data = torch.tensor([[2.0], [4.0], [6.0]])

#基本都把模型定义成一个类
class LinearModel(torch.nn.Module):#必须继承torch.nn.Module
    #至少实现两个函数 构造函数 前向过程
    def __init__(self):
        super(LinearModel, self).__init__()
        # (1,1)是指输入x和输出y的特征维度，这里数据集中的x和y的特征都是1维的 有这两个可以算出w和b的维度
        # 该线性层需要学习的参数是w和b  获取w/b的方式分别是~linear.weight/linear.bias
        self.linear = torch.nn.Linear(1, 1,bias=True)
        print('初始权重',self.linear.weight.item())
        print('初始偏移',self.linear.bias.item())
    # #     初始化是随机的 尝试直接初始化赋值了一下 区分.data和.item
    #     self.linear.weight.data=torch.tensor([[2.0]],requires_grad=True)
    #     self.linear.bias.data=torch.tensor([0.0],requires_grad=True)

    def forward(self, x):
        y_pred = self.linear(x)#overwrite
        return y_pred

#实例化模型 callable的
model = LinearModel()

# construct loss (MSE）
# criterion = torch.nn.MSELoss(size_average = False) 不做1/N LOSS
criterion = torch.nn.MSELoss(reduction='sum')

# construct  optimizer(SGD)
# model.parameters()检查模型中的参数 将其全部设置为优化的对象
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)  # 优化器 model.parameters()自动完成参数的初始化操作  相当于给参数requireguard=True lr是权重学习率

# 训练过程
for epoch in range(100):
    y_pred = model(x_data)  # forward:predict
    loss = criterion(y_pred, y_data)  # forward计算loss
    print(epoch, loss.item())
    optimizer.zero_grad()  # the grad computer by .backward() will be accumulated. so before backward, remember set the grad to zero
    loss.backward()  # backward: autograd，自动计算梯度
    optimizer.step()  # update 参数，即更新w和b的值

print('w = ', model.linear.weight.item())
print('b = ', model.linear.bias.item())

x_test = torch.tensor([[4.0]]) #测试集
y_test = model(x_test)
print('y_pred = ', y_test.data)
