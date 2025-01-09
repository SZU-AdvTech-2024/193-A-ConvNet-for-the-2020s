# """
# 视频教程：https://www.bilibili.com/video/BV1p7411T7Pc/?spm_id_from=333.788
# flower数据集为5分类数据集，共有 {'daisy':0, 'dandelion':1, 'roses':2, 'sunflower':3, 'tulips':4} 5个分类。
#
# 该程序用于将数据集切分为训练集和验证集，使用步骤如下：
# （1）在"split_data.py"的同级路径下创建新文件夹"flower_data"
# （2）点击链接下载花分类数据集 http://download.tensorflow.org/example_images/flower_photos.tgz
# （3）解压数据集到flower_data文件夹下
# （4）执行"split_data.py"脚本自动将数据集划分为训练集train和验证集val
#
# 切分后的数据集结构：
# ├── split_data.py
# ├── flower_data
#        ├── flower_photos.tgz （下载的未解压的原始数据集）
#        ├── flower_photos（解压的数据集文件夹，3670个样本）
#        ├── train（生成的训练集，3306个样本）
#        └── val（生成的验证集，364个样本）
# """""
#
#
# import os
# from shutil import copy, rmtree
# import random
#
#
# def mk_file(file_path: str):
#     if os.path.exists(file_path):
#         # 如果文件夹存在，则先删除原文件夹在重新创建
#         rmtree(file_path)
#     os.makedirs(file_path)
#
#
# def main():
#     random.seed(0)
#
#     # 将数据集中10%的数据划分到验证集中
#     split_rate = 0.1
#
#     # 指向你解压后的flower_photos文件夹
#     cwd = os.getcwd()
#     data_path = os.path.join(cwd, "E:/Code/data/flower_photos")
#     data_root=os.path.join(cwd, "flower_data")
#     origin_flower_path = os.path.join(data_path, "")
#     assert os.path.exists(origin_flower_path), "path '{}' does not exist.".format(origin_flower_path)
#
#     flower_class = [cla for cla in os.listdir(origin_flower_path)
#                     if os.path.isdir(os.path.join(origin_flower_path, cla))]
#
#     # 建立保存训练集的文件夹
#     train_root = os.path.join(data_root, "train")
#     mk_file(train_root)
#     for cla in flower_class:
#         # 建立每个类别对应的文件夹
#         mk_file(os.path.join(train_root, cla))
#
#     # 建立保存验证集的文件夹
#     val_root = os.path.join(data_root, "val")
#     mk_file(val_root)
#     for cla in flower_class:
#         # 建立每个类别对应的文件夹
#         mk_file(os.path.join(val_root, cla))
#
#     for cla in flower_class:
#         cla_path = os.path.join(origin_flower_path, cla)
#         images = os.listdir(cla_path)
#         num = len(images)
#         # 随机采样验证集的索引
#         eval_index = random.sample(images, k=int(num*split_rate))
#         for index, image in enumerate(images):
#             if image in eval_index:
#                 # 将分配至验证集中的文件复制到相应目录
#                 image_path = os.path.join(cla_path, image)
#                 new_path = os.path.join(val_root, cla)
#                 copy(image_path, new_path)
#             else:
#                 # 将分配至训练集中的文件复制到相应目录
#                 image_path = os.path.join(cla_path, image)
#                 new_path = os.path.join(train_root, cla)
#                 copy(image_path, new_path)
#             print("\r[{}] processing [{}/{}]".format(cla, index+1, num), end="")  # processing bar
#         print()
#
#     print("processing done!")
#
#
# if __name__ == '__main__':
#     main()


#split_data.py
#划分数据集flower_data，数据集划分到flower_datas中，训练集：验证集：测试集比例为6：2：2

import os
import random
from shutil import copy2

# 源文件路径
file_path = r"E:/Code/data/flower_photos"
# 新文件路径
new_file_path = r"E:/Code/data/flower_photo"
# 划分数据比例为6:2:2
split_rate = [0.6, 0.2, 0.2]
print("Starting...")
print("Ratio= {}:{}:{}".format(int(split_rate[0] * 10), int(split_rate[1] * 10), int(split_rate[2] * 10)))
class_names = os.listdir(file_path)
# 在目标目录下创建文件夹
split_names = ['train', 'val', 'test']
# 判断是否存在木匾文件夹
if os.path.isdir(new_file_path):
    pass
else:
    os.mkdir(new_file_path)
for split_name in split_names:
    # split_path = os.path.join(new_file_path, split_name)
    split_path = new_file_path + "/" + split_name
    if os.path.isdir(split_path):
        pass
    else:
        os.mkdir(split_path)
    # 然后在split_path的目录下创建类别文件夹
    for class_name in class_names:
        class_split_path = os.path.join(split_path, class_name)
        if os.path.isdir(class_split_path):
            pass
        else:
            os.mkdir(class_split_path)

# 按照比例划分数据集，并进行数据图片的复制
# 首先进行分类遍历
for class_name in class_names:
    current_class_data_path = os.path.join(file_path, class_name)
    current_all_data = os.listdir(current_class_data_path)
    current_data_length = len(current_all_data)
    current_data_index_list = list(range(current_data_length))
    random.shuffle(current_data_index_list)

    train_path = os.path.join(os.path.join(new_file_path, 'train'), class_name)
    val_path = os.path.join(os.path.join(new_file_path, 'val'), class_name)
    test_path = os.path.join(os.path.join(new_file_path, 'test'), class_name)
    train_stop_flag = current_data_length * split_rate[0]
    val_stop_flag = current_data_length * (split_rate[0] + split_rate[1])
    current_idx = 0
    train_num = 0
    val_num = 0
    test_num = 0
    for i in current_data_index_list:
        src_img_path = os.path.join(current_class_data_path, current_all_data[i])
        if current_idx <= train_stop_flag:
            copy2(src_img_path, train_path)
            train_num = train_num + 1
        elif (current_idx > train_stop_flag) and (current_idx <= val_stop_flag):
            copy2(src_img_path, val_path)
            val_num = val_num + 1
        else:
            copy2(src_img_path, test_path)
            test_num = test_num + 1

        current_idx = current_idx + 1

    print("<{}> has {} pictures,train:val:test={}:{}:{}".format(class_name, current_data_length, train_num, val_num,
                                                              test_num))
print("Done")



