# 学习transforms 可以把图片or numpy变tensor 还有resize norm
# PIL Image.open() tensor ToTensor() narrays cv.imread()
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import os

from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
writer =SummaryWriter("logs")
class MyData(Dataset):

    def __init__(self, root_dir, image_dir, label_dir, transform=None):
        self.root_dir = root_dir
        self.image_dir = image_dir
        self.label_dir = label_dir
        self.label_path = os.path.join(self.root_dir, self.label_dir)
        self.image_path = os.path.join(self.root_dir, self.image_dir)
        self.image_list = os.listdir(self.image_path)
        self.label_list = os.listdir(self.label_path)
        self.transform = transform
        # 因为label 和 Image文件名相同，进行一样的排序，可以保证取出的数据和label是一一对应的
        self.image_list.sort()
        self.label_list.sort()

    def __getitem__(self, idx):
        img_name = self.image_list[idx]
        label_name = self.label_list[idx]
        img_item_path = os.path.join(self.root_dir, self.image_dir, img_name)
        label_item_path = os.path.join(self.root_dir, self.label_dir, label_name)
        img = Image.open(img_item_path)
        with open(label_item_path, 'r') as f:
            label = f.readline()

        if self.transform:
            img = transform(img)


        return img, label

    def __len__(self):
        assert len(self.image_list) == len(self.label_list)
        return len(self.image_list)

# Compose Resize Totensor
transform = transforms.Compose([transforms.Resize(400), transforms.ToTensor()])
root_dir = "data/train"
image_ants = "ants_image"
label_ants = "ants_label"
ants_dataset = MyData(root_dir, image_ants, label_ants, transform=transform)
image_bees = "bees_image"
label_bees = "bees_label"
bees_dataset = MyData(root_dir, image_bees, label_bees, transform=transform)

# # ToTensor
# img_tensor,label=bees_dataset.__getitem__(2)
# writer.add_image("ToTensor",img_tensor)
# writer.close()
# # tensorboard --logdir=E:\program\python_project\study_Pytorch\day3\logs
# print(type(img_tensor))

# #Normalize
# #均值 标准差
# print(img_tensor[0][0][0])
# trans_norm=transforms.Normalize([0.5,0.5,0.5],[0.5,0.5,0.5])
# img_norm=trans_norm(img_tensor)
# print(img_norm[0][0][0])
# writer.add_image("Normalize",img_norm)

# #resize
# test_dataset = MyData(root_dir, image_ants, label_ants)
# img,label=test_dataset.__getitem__(1)
# print(img.size)
# transform = transforms.Resize((100,100))
# img=transform(img)
# print(img.size)
# img.show()


