# 学习如何读取数据库
from torch.utils.data import Dataset
from PIL import Image
import os
if __name__ == '__main__':
    # # 查看一个函数极其教程

    # print(dir(torch))
    # # __x__私有属性 无法篡改
    # print(dir(torch.cuda.is_available))
    # help(torch.cuda.is_available)

    #如何读取数据
    class Mydata(Dataset):

        def __init__(self,root_dir,label_dir):
            self.root_dir=root_dir
            self.label_dir=label_dir
            self.path=os.path.join(self.root_dir,self.label_dir)
            self.img_path=os.listdir(self.path)

        def __getitem__(self, idx):
            # img_path="dataset/train/ants/0013035.jpg"
            # img=Image.open(img_path)
            # print(img.size)
            # img.show()
            #
            # #整个文件夹的图片
            # dataset_path = "dataset/train/ants"
            # img_path_list=os.listdir(dataset_path)
            # print(img_path_list[0])

            image_name=self.img_path[idx]
            img_item_path=os.path.join(self.path,image_name)
            img =Image.open(img_item_path)
            label=self.label_dir
            return img,label

        def __len__(self):
            return len(self.img_path)

root_dir="dataset/train"
ants_label_dir="ants"
ants_dataset=Mydata(root_dir,ants_label_dir)
print(ants_dataset[0])
img,lable=ants_dataset[0]

bees_label_dir="bees"
bees_dataset=Mydata(root_dir,bees_label_dir)
train_dataset=ants_dataset+bees_dataset
print(range(len(train_dataset)))
for i in range(len(train_dataset)):
    print(train_dataset.__getitem__(i))

