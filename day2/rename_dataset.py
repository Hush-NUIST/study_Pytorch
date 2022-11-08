import os

root_dir = 'dataset/train'
target_dir = 'ants'
# os.listdir() 方法用于返回指定的文件夹包含的文件或文件夹的名字的列表
img_path = os.listdir(os.path.join(root_dir, target_dir))
# Python split() 通过指定分隔符对字符串进行切片，如果参数 num 有指定值，则分隔 num+1 个子字符串
# 返回分割后的字符串列表。
# str.split(“o”)[0]得到的是第一个o之前的内容
# str.split(“o”)[1]得到的是第一个o和第二个o之间的内容
label = target_dir.split('_')[0]
out_dir = 'ants_label'
for i in img_path:
    file_name = i.split('.jpg')[0]
    with open(os.path.join(root_dir, out_dir,"{}.txt".format(file_name)),'w') as f:
        f.write(label)

# 自己试一下
target_dir2 = 'bees'
img_path2 = os.listdir(os.path.join(root_dir, target_dir2))
label2 = target_dir2.split('_')[0]
out_dir2 = 'bees_label'
for i in img_path2:
    file_name2 = i.split('.jpg')[0]
    with open(os.path.join(root_dir, out_dir2,"{}.txt".format(file_name2)),'w') as f:
        f.write(label2)
