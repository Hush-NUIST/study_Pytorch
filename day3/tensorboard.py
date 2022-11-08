# 学习tensorboard
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from PIL import Image
if __name__ == '__main__':
    writer= SummaryWriter("logs")
    #图片
    image_path="data/train/ants_image/0013035.jpg"
    image_path2 = "data/train/bees_image/16838648_415acd9e3f.jpg"
    img=Image.open(image_path)
    img2 = Image.open(image_path2)
    print(type(img))
    #numpy型一般是用opencv
    img_array=np.array(img)
    img_array2 = np.array(img2)
    #  writer.add_image输入必须是numpy或者tensor
    print(type(img_array))
    print(img_array.shape)
    writer.add_image("test",img_array,1,dataformats='HWC')
    writer.add_image("test", img_array2, 2, dataformats='HWC')

    #函数
    for i in range(100):
      writer.add_scalar("y=2x",2*i,i)
    writer.close()

# 然后我在conda里面跑tensorboard --logdir=E:\program\python_project\study_Pytorch\day3\logs --port=6006 不知道为啥terminal跑不了

