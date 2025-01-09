import os

def generate_labels(data_dir, output_file):
    # 创建或打开 label.txt 文件
    with open(output_file, 'w') as f:
        # 遍历验证集目录
        for class_name in os.listdir(data_dir):
            class_folder = os.path.join(data_dir, class_name)
            if os.path.isdir(class_folder):  # 如果是文件夹，表示是一个类
                for img_name in os.listdir(class_folder):
                    img_path = os.path.join(class_folder, img_name)
                    if img_name.endswith('.jpg'):  # 只处理 .jpg 文件
                        # 写入文件名和类名
                        f.write(f"{img_name} {class_name}\n")

# 示例：调用函数生成标签文件
data_dir = 'ConvNeXt/flower_data/val'  # 这里替换为验证集文件夹路径
output_file = 'ConvNeXt/flower_data/plot_img/label.txt'  # 输出的标签文件路径

generate_labels(data_dir, output_file)
print(f"标签文件 {output_file} 已生成.")
