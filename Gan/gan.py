import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import torchvision
from torch.utils.tensorboard import SummaryWriter, writer
from torchvision import transforms, datasets

# ******************************************************准备数据集********************************************************
transform = transforms.Compose([
    transforms.ToTensor(),  # 归一化为0~1
    transforms.Normalize(0.5, 0.5)  # 归一化为-1~1
])
train_ds = datasets.MNIST(root='./dataset/mnist_train/', train=True, download=False, transform=transform)
dataloader = torch.utils.data.DataLoader(train_ds, batch_size=64, shuffle=True)
imgs, _ = next(iter(dataloader))


# ******************************************************搭建模型*********************************************************
# 生成器
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()  # 继承父类
        self.main = nn.Sequential(
            nn.Linear(100, 256), nn.ReLU(),
            nn.Linear(256, 512), nn.ReLU(),
            nn.Linear(512, 28 * 28),
            nn.Tanh()  # 最后必须用tanh，把数据分布到（-1，1）之间
        )

    def forward(self, x):  # x表示长度为100的噪声输入
        img = self.main(x)
        img = img.view(-1, 1,28, 28)  # 方便等会绘图
        return img


# 判别器
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.main = nn.Sequential(
            nn.Linear(28 * 28, 512),
            nn.LeakyReLU(),  # x小于零是是一个很小的值不是0，x大于0是还是x
            nn.Linear(512, 256),
            nn.LeakyReLU(),
            nn.Linear(256, 1),
            nn.Sigmoid()  # 保证输出范围为（0，1）的概率
        )

    def forward(self, x):  # x表示28*28的mnist图片
        img = x.view(-1, 28 * 28)
        img = self.main(img)
        return img


# 实例化
device = 'cuda' if torch.cuda.is_available() else 'cpu'
# 模型
G = Generator().to(device)
D = Discriminator().to(device)
# 优化器
g_opt = torch.optim.Adam(G.parameters(), lr=0.0001)
d_opt = torch.optim.Adam(D.parameters(), lr=0.0001)
# 损失
loss = torch.nn.BCELoss()

test_input = torch.randn(16, 100, device=device)


def train(epochs):
    k = 1
    for epoch in range(epochs):
        d_epoch_loss = 0
        g_epoch_loss = 0
        count = len(dataloader)  # 一个epoch的大小
        # sample minibatch
        for _, (img, _) in enumerate(dataloader):
            img = img.to(device)  # 一个批次的图片
            size = img.size(0)  # 和图片对应的原始噪音
            random_noise = torch.randn(size, 100, device=device)
            # 训练K次判别器
            for step in range(k):
                gen_img = G(random_noise)  # 生成的图像
                d_opt.zero_grad()
                real_output = D(img)  # 判别器输入真实图片，对真实图片的预测结果，希望是1
                # 判别器在真实图像上的损失
                d_real_loss = loss(real_output, torch.ones_like(real_output))  # size一样全一的tensor
                d_real_loss.backward()
                # 冻结生成器
                g_opt.zero_grad()
                # .detach的效果等同于requires_grad置为False.
                fake_output = D(gen_img.detach())  # 判别器输入生成图片，对生成图片的预测结果，希望是0
                # 判别器在生成图像上的损失
                d_fake_loss = loss(fake_output, torch.zeros_like(fake_output))  # size一样全一的tensor
                d_fake_loss.backward()
                d_loss = d_real_loss + d_fake_loss
                d_opt.step()
                with torch.no_grad():
                    d_epoch_loss += d_loss

            # 训练1次生成器
            g_opt.zero_grad()
            fake_output = D(gen_img)  # 希望被判定为1
            g_loss = loss(fake_output, torch.ones_like(fake_output))
            g_loss.backward()
            g_opt.step()
            with torch.no_grad():
                g_epoch_loss += g_loss
        # 一个epoch训练完成
        print('第', epoch + 1, '个epoch 训练判别器的loss为', d_epoch_loss)
        print('第', epoch + 1, '个epoch 训练生成器的loss为', g_epoch_loss)
        writer = SummaryWriter("./logs_GAN")
        writer.add_images("generate", G(test_input).cpu(), epoch)
        # gen_img_plot(G, test_input)

# tensorboard ––logdir=E:\program\python_project\adversarial\mycode\logs_GAN
if __name__ == '__main__':
    train(50)
