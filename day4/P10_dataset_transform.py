# 学习直接下载数据集
import torchvision
from torch.utils.tensorboard import SummaryWriter

dataset_transform = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor()
])
# 下载数据集 train=true训练集 download=True下载 想转tensor或者resize可以直接 transform=xxx
train_set = torchvision.datasets.CIFAR10(root="./dataset", train=True, transform=dataset_transform, download=False)
test_set = torchvision.datasets.CIFAR10(root="./dataset", train=False, transform=dataset_transform, download=False)
PIL_set=torchvision.datasets.CIFAR10(root="./dataset", train=False, download=False)

print(PIL_set[0])
print(PIL_set.classes)
img, target = PIL_set[0]
print(img)
print(target)
print(PIL_set.classes[target])
img.show()
print(test_set[0])

writer = SummaryWriter("p10")
for i in range(10):
    img, target = test_set[i]
    writer.add_image("test_set", img, i)
#  tensorboard --logdir=E:\program\python_project\study_Pytorch\day4\p10
writer.close()