# 学习pytorch基本操作和概念
import torch
import numpy as np

if __name__ == '__main__':
    # print(torch.__version__)
    # print(torch.cuda.is_available())



    # # 创建一个tensor
    # x=torch.empty(5,3)
    # print(x)
    # x=torch.rand(5,3)
    # print(x)
    # x=torch.tensor([5.5,3])
    # print(x,x.size())




    # # view操作可以改变矩阵维度
    # x=torch.rand(4,4)
    # print(x)
    # y=x.view(16)
    # print(y)
    # # -1自动去做计算
    # z=x.view(-1,8)
    # print(z)



    # tensor可以鱼numpy协同操作与转换
    # a=torch.rand(1,4)
    # b=a.numpy()
    # print(b)
    # a=np.random(4)
    # b=torch.from_numpy(a)
    # print(a)
    # print(b)


# tensor(data, dtype=None, device=None, requires_grad=False) -> Tensor
#
# 参数:
# data： (array_like): tensor的初始值.可以是列表，元组，numpy数组，标量等;
# dtype： tensor元素的数据类型
# device： 指定CPU或者是GPU设备，默认是None
# requires_grad：是否可以求导，即求梯度，默认是False，即不可导的

    # # 自动计算反向传播机制
    # x1=torch.randn(3,4,requires_grad=True)
    # b=torch.randn(3,4,requires_grad=True)
    # t=x1+b
    # y=t.sum()
    # # 求偏导? 对 因为b和x1都是可导的
    # y.backward()
    # print(x1.grad)
    # print(b.grad)
    # print(x1.requires_grad,b.requires_grad,t.requires_grad)

    # # 搞个简单点的易理解
    # x2=torch.tensor(3,dtype=float,requires_grad=True)
    # y=torch.pow(x2,2)
    # # y=x^2求导且x=3
    # y.backward()
    # print(x2.grad)

    # # 视频题目
    # x=torch.randn(1)
    # b=torch.randn(1,requires_grad=True)
    # w=torch.randn(1,requires_grad=True)
    # y=w*x
    # z=y+b
    # # 有了这个参数才可以两次反向传播 两次传播梯度相加
    # z.backward(retain_graph=True)
    # z.backward()
    # print(x)
    # print(w.grad,b.grad)

    #一个线性神经网络模型 反向传播
    x_data = [1.0, 2.0, 3.0] #训练集
    y_data = [2.0, 4.0, 6.0]
    w = torch.tensor([1.0])  # 权重的初值为1.0
    w.requires_grad = True  # 需要计算梯度


    def forward(x):
        return x * w  #前向过程


    def loss(x, y):
        y_pred = forward(x)
        return (y_pred - y) ** 2 #损失函数 最小二乘法


    print("predict (before training)", 4, forward(4).item()) #4是测试集

    for epoch in range(100):
        for x, y in zip(x_data, y_data):
            l = loss(x, y)  # l是一个张量，tensor主要是在建立计算图 forward, compute the loss
            l.backward()  # 相当于从后往前一步一步求出前向过程中的所有梯度
            print('\tgrad:', x, y, w.data)
            w.data = w.data - 0.01 * w.grad.data  # 更新权重参数 。data标量
            w.grad.data.zero_()  # 不清0 会出现75行的情况

        print('progress:', epoch,'损失为', l.item())  # 取出loss使用l.item，不要直接使用l（l是tensor会构建计算图）

    print("predict (after training)", 4, forward(4).item())
