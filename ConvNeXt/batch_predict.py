import os
import json

import torch
from PIL import Image
from torchvision import transforms

from model import convnext_tiny


def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    img_size = 224
    data_transform = transforms.Compose(
        [transforms.Resize(int(img_size * 1.14)),
         transforms.CenterCrop(img_size),
         transforms.ToTensor(),
         transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    # load image
    # 指向需要遍历预测的图像文件夹
    imgs_root = r"D:/other/ClassicalModel/data/flower_datas/train/daisy"
    assert os.path.exists(imgs_root), f"file: '{imgs_root}' dose not exist."
    # 读取指定文件夹下所有jpg图像路径
    img_path_list = [os.path.join(imgs_root, i) for i in os.listdir(imgs_root) if i.endswith(".jpg")]

    # read class_indict
    json_path = r"D:/other/ClassicalModel/ResNet/class_indices.json"
    assert os.path.exists(json_path), f"file: '{json_path}' dose not exist."

    json_file = open(json_path, "r")
    class_indict = json.load(json_file)

    # create model
    model = convnext_tiny(num_classes=5).to(device)

    # load model weights
    weights_path = r"D:/other/ClassicalModel/ConvNeXt/runs1/weights/bestmodel.pth"
    assert os.path.exists(weights_path), f"file: '{weights_path}' dose not exist."
    model.load_state_dict(torch.load(weights_path, map_location=device))

    #save predicted img
    filename = 'record.txt'
    save_path = 'detect'
    path_num = 1
    while os.path.exists(save_path + f'{path_num}'):
        path_num += 1
    os.mkdir(save_path + f'{path_num}')
    f = open(save_path + f'{path_num}/' + filename, 'w')
    f.write("imgs_root:"+imgs_root+"\n")
    f.write("weights_path:"+weights_path+"\n")

    actual_classes="daisy"
    acc_num=0
    all_num=len(img_path_list)
    # prediction
    model.eval()
    batch_size = 8  # 每次预测时将多少张图片打包成一个batch
    with torch.no_grad():
        for ids in range(0, len(img_path_list) // batch_size):
            img_list = []
            for img_path in img_path_list[ids * batch_size: (ids + 1) * batch_size]:
                assert os.path.exists(img_path), f"file: '{img_path}' dose not exist."
                img = Image.open(img_path)
                img = data_transform(img)
                img_list.append(img)

            # batch img
            # 将img_list列表中的所有图像打包成一个batch
            batch_img = torch.stack(img_list, dim=0)
            # predict class
            output = model(batch_img.to(device)).cpu()
            predict = torch.softmax(output, dim=1)
            probs, classes = torch.max(predict, dim=1)

            for idx, (pro, cla) in enumerate(zip(probs, classes)):
                print("image: {}  class: {}  prob: {:.3}".format(img_path_list[ids * batch_size + idx],
                                                                 class_indict[str(cla.numpy())],
                                                                 pro.numpy()))
                f.write("image: {}  class: {}  prob: {:.3}\n".format(img_path_list[ids * batch_size + idx],
                                                                 class_indict[str(cla.numpy())],
                                                                 pro.numpy()))
                if class_indict[str(cla.numpy())]==actual_classes:
                    acc_num+=1
    print("classes:{},acc_num:{:d},all_num:{:d},accuracy: {:.3f}".format(actual_classes,acc_num,all_num,acc_num/all_num))
    f.write("classes:{},acc_num:{:d},all_num:{:d},accuracy: {:.3f}".format(actual_classes,acc_num,all_num,acc_num/all_num))
    f.close()
if __name__ == '__main__':
    main()

