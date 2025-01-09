# import json
# import os
# import argparse
# import time
#
# import torch
# import torch.optim as optim
# from torch.utils.tensorboard import SummaryWriter
# from torchvision import transforms, datasets
#
# from model import convnext_tiny as create_model
# from utils import  create_lr_scheduler, get_params_groups, train_one_epoch, evaluate,plot_class_preds
#
#
# def main(args):
#     os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
#     device = torch.device(args.device if torch.cuda.is_available() else "cpu")
#
#     print(args)
#     # 创建定义文件夹以及文件
#     filename = 'record.txt'
#     save_path = 'runs'
#     path_num = 1
#     while os.path.exists(save_path + f'{path_num}'):
#         path_num += 1
#     save_path = save_path + f'{path_num}'
#     os.mkdir(save_path)
#     f = open(save_path + "/" + filename, 'w')
#     f.write("{}\n".format(args))
#
#     # print('Start Tensorboard with "tensorboard --logdir=runs", view at http://localhost:6006/')
#     # 实例化SummaryWriter对象
#     # #######################################
#     tb_writer = SummaryWriter(log_dir=save_path + "/flower_experiment")
#     if os.path.exists(save_path + "/weights") is False:
#         os.makedirs(save_path + "/weights")
#
#     img_size = 224
#     data_transform = {
#         "train": transforms.Compose([transforms.RandomResizedCrop(img_size),
#                                      transforms.RandomHorizontalFlip(),
#                                      transforms.ToTensor(),
#                                      transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]),
#         "val": transforms.Compose([transforms.Resize(int(img_size * 1.143)),
#                                    transforms.CenterCrop(img_size),
#                                    transforms.ToTensor(),
#                                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])}
#
#     # 实例化训练数据集
#     train_data_set = datasets.ImageFolder(root=os.path.join(args.data_path, "train"),
#                                           transform=data_transform["train"])
#
#     # 实例化验证数据集
#     val_data_set = datasets.ImageFolder(root=os.path.join(args.data_path, "val"),
#                                         transform=data_transform["val"])
#
#     # 生成class_indices.json文件，包括有模型对应的序列号
#     # #######################################
#     classes_list = train_data_set.class_to_idx
#     cla_dict = dict((val, key) for key, val in classes_list.items())
#     # write dict into json file
#     json_str = json.dumps(cla_dict, indent=4)
#     with open('class_indices.json', 'w') as json_file:
#         json_file.write(json_str)
#
#     batch_size = args.batch_size
#     nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])  # number of workers
#     print('Using {} dataloader workers every process'.format(nw))
#     train_loader = torch.utils.data.DataLoader(train_data_set,
#                                                batch_size=batch_size,
#                                                shuffle=True,
#                                                pin_memory=True,
#                                                num_workers=nw)
#
#     val_loader = torch.utils.data.DataLoader(val_data_set,
#                                              batch_size=batch_size,
#                                              shuffle=False,
#                                              pin_memory=True,
#                                              num_workers=nw)
#
#     model = create_model(num_classes=args.num_classes).to(device)
#
#     # Write the model into tensorboard
#     # #######################################
#     init_img = torch.zeros((1, 3, 224, 224), device=device)
#     tb_writer.add_graph(model, init_img)
#
#     if args.weights != "":
#         assert os.path.exists(args.weights), "weights file: '{}' not exist.".format(args.weights)
#         # weights_dict = torch.load(args.weights, map_location=device)
#         weights_dict = torch.load(args.weights, map_location=device)["model"]
#         # 删除有关分类类别的权重
#         for k in list(weights_dict.keys()):
#             if "head" in k:
#                 del weights_dict[k]
#         print(model.load_state_dict(weights_dict, strict=False))
#
#     if args.freeze_layers:
#         for name, para in model.named_parameters():
#             # 除head外，其他权重全部冻结
#             if "head" not in name:
#                 para.requires_grad_(False)
#             else:
#                 print("training {}".format(name))
#
#     # pg = [p for p in model.parameters() if p.requires_grad]
#     pg = get_params_groups(model, weight_decay=args.wd)
#     optimizer = optim.AdamW(pg, lr=args.lr, weight_decay=args.wd)
#     lr_scheduler = create_lr_scheduler(optimizer, len(train_loader), args.epochs,
#                                        warmup=True, warmup_epochs=1)
#
#     best_acc = 0.0
#     for epoch in range(args.epochs):
#         # 计时器time_start
#         time_start = time.time()
#         # train
#         train_loss, train_acc = train_one_epoch(model=model,
#                                                 optimizer=optimizer,
#                                                 data_loader=train_loader,
#                                                 device=device,
#                                                 epoch=epoch,
#                                                 lr_scheduler=lr_scheduler)
#
#         # validate
#         val_loss, val_acc = evaluate(model=model,
#                                      data_loader=val_loader,
#                                      device=device,
#                                      epoch=epoch)
#         time_end = time.time()
#         f.write("[epoch {}] train_loss: {:.3f},train_acc:{:.3f},val_loss:{:.3f},val_acc:{:.3f},Spend_time:{:.3f}S"
#                 .format(epoch + 1, train_loss, train_acc, val_loss, val_acc, time_end - time_start))
#         f.flush()
#
#         # add Training results into tensorboard
#         # #######################################
#         tags = ["train_loss", "train_acc", "val_loss", "val_acc", "learning_rate"]
#         tb_writer.add_scalar(tags[0], train_loss, epoch)
#         tb_writer.add_scalar(tags[1], train_acc, epoch)
#         tb_writer.add_scalar(tags[2], val_loss, epoch)
#         tb_writer.add_scalar(tags[3], val_acc, epoch)
#         tb_writer.add_scalar(tags[4], optimizer.param_groups[0]["lr"], epoch)
#
#         # add figure into tensorboard
#         # #######################################
#         fig = plot_class_preds(net=model,
#                                images_dir=r"flower_data/plot_img",
#                                transform=data_transform["val"],
#                                num_plot=6,
#                                device=device)
#         if fig is not None:
#             tb_writer.add_figure("predictions vs. actuals",
#                                  figure=fig,
#                                  global_step=epoch)
#
#         if val_acc > best_acc:
#             best_acc = val_acc
#             f.write(',save best model')
#             torch.save(model.state_dict(), save_path + "/weights/bestmodel.pth")
#         f.write('\n')
#     f.close()
#
# if __name__ == '__main__':
#     parser = argparse.ArgumentParser()
#     parser.add_argument('--num_classes', type=int, default=5)
#     parser.add_argument('--epochs', type=int, default=2)
#     parser.add_argument('--batch-size', type=int, default=4)
#     parser.add_argument('--lr', type=float, default=5e-4)
#     parser.add_argument('--wd', type=float, default=5e-2)
#     parser.add_argument('--data-path', type=str,
#                         default=r"E:/Code/ConvNeXt-main/ConvNeXt/flower_data/")
#     parser.add_argument('--weights', type=str,
#                         default=r"E:/Code/ConvNeXt-main/convnext_tiny_22k_1k_224.pth",
#                         help='initial weights path')
#     # 是否冻结head以外所有权重
#     parser.add_argument('--freeze-layers', type=bool, default=False)
#     parser.add_argument('--device', default='cuda:0', help='device id (i.e. 0 or 0,1 or cpu)')
#
#     opt = parser.parse_args()
#
#     main(opt)
#


import torch
import numpy as np
import matplotlib.pyplot as plt
import json
import os
import argparse
import time

import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms, datasets

from model import convnext_tiny as create_model
from utils import create_lr_scheduler, get_params_groups, train_one_epoch, evaluate, plot_class_preds


def generate_label_file(data_dir, output_file):
    """
    生成验证集的标签文件，格式：filename.jpg class_name
    """
    with open(output_file, 'w') as f:
        for class_name in os.listdir(data_dir):
            class_folder = os.path.join(data_dir, class_name)
            if os.path.isdir(class_folder):  # 如果是文件夹，表示一个类别
                for img_name in os.listdir(class_folder):
                    img_path = os.path.join(class_folder, img_name)
                    if img_name.endswith('.jpg'):  # 确保是 jpg 文件
                        f.write(f"{img_name} {class_name}\n")  # 写入标签文件

def log_training_results(tb_writer, epoch, train_loss, train_acc, val_loss, val_acc, lr, plot_every=1, train_losses=None, val_losses=None, train_accuracies=None, val_accuracies=None):
    """
    记录训练和验证损失、准确度，并将其显示在 TensorBoard 中。

    Parameters:
    - tb_writer: TensorBoard writer
    - epoch: 当前 epoch
    - train_loss: 当前 epoch 训练集的损失
    - train_acc: 当前 epoch 训练集的准确率
    - val_loss: 当前 epoch 验证集的损失
    - val_acc: 当前 epoch 验证集的准确率
    - lr: 当前学习率
    - plot_every: 记录频率
    - train_losses: 用于记录训练损失的列表
    - val_losses: 用于记录验证损失的列表
    - train_accuracies: 用于记录训练准确率的列表
    - val_accuracies: 用于记录验证准确率的列表
    """

    # 更新训练和验证的损失和准确度列表
    train_losses.append(train_loss)
    val_losses.append(val_loss)
    train_accuracies.append(train_acc)
    val_accuracies.append(val_acc)

    # 记录每个 epoch 的损失和准确率
    tb_writer.add_scalar("train_loss", train_loss, epoch)
    tb_writer.add_scalar("train_acc", train_acc, epoch)
    tb_writer.add_scalar("val_loss", val_loss, epoch)
    tb_writer.add_scalar("val_acc", val_acc, epoch)
    tb_writer.add_scalar("learning_rate", lr, epoch)

    if epoch % plot_every == 0:
        # 绘制训练和验证损失折线图
        fig, ax = plt.subplots()
        ax.plot(range(len(train_losses)), train_losses, label="Train Loss", color='blue')
        ax.plot(range(len(val_losses)), val_losses, label="Val Loss", color='red')
        ax.set_xlabel('Epochs')
        ax.set_ylabel('Loss')
        ax.legend()
        tb_writer.add_figure("Loss over epochs", fig, global_step=epoch)

        # 绘制训练和验证准确率折线图
        fig, ax = plt.subplots()
        ax.plot(range(len(train_accuracies)), train_accuracies, label="Train Accuracy", color='blue')
        ax.plot(range(len(val_accuracies)), val_accuracies, label="Val Accuracy", color='red')
        ax.set_xlabel('Epochs')
        ax.set_ylabel('Accuracy')
        ax.legend()
        tb_writer.add_figure("Accuracy over epochs", fig, global_step=epoch)

def log_validation_results(tb_writer, model, val_loader, epoch, device, num_samples=6):
    """
    生成并记录验证集的预测结果与真实标签对比图到 TensorBoard。

    Parameters:
    - tb_writer: TensorBoard writer
    - model: 训练好的模型
    - val_loader: 验证集的 dataloader
    - epoch: 当前 epoch
    - device: 训练使用的设备（CPU 或 GPU）
    - num_samples: 每次绘制的样本数量
    """
    model.eval()
    images = []
    labels = []
    preds = []

    with torch.no_grad():
        for i, (inputs, target) in enumerate(val_loader):
            if i >= num_samples:  # 只绘制 num_samples 张图片
                break
            inputs, target = inputs.to(device), target.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)

            # 收集图片、真实标签和预测标签
            images.append(inputs.cpu())
            labels.append(target.cpu())
            preds.append(predicted.cpu())

    # 将图片拼接成一个网格
    images = torch.cat(images, dim=0)
    labels = torch.cat(labels, dim=0)
    preds = torch.cat(preds, dim=0)

    # 绘制结果
    fig = plt.figure(figsize=(12, 12))
    for i in range(num_samples):
        ax = fig.add_subplot(2, num_samples // 2, i + 1)
        img = images[i].permute(1, 2, 0).numpy()  # 转换成 HWC 格式
        img = np.clip(img, 0, 1)  # 归一化到 [0, 1] 区间
        ax.imshow(img)
        ax.axis('off')
        ax.set_title(f"True: {labels[i].item()}, Pred: {preds[i].item()}")

    # 将图像添加到 TensorBoard
    tb_writer.add_figure("Validation Predictions vs Actuals", fig, epoch)
    model.train()


def main(args):
    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    print(args)
    filename = 'record.txt'
    save_path = 'runs'
    path_num = 1
    while os.path.exists(save_path + f'{path_num}'):
        path_num += 1
    save_path = save_path + f'{path_num}'
    os.mkdir(save_path)
    f = open(save_path + "/" + filename, 'w')
    f.write("{}\n".format(args))

    tb_writer = SummaryWriter(log_dir=save_path + "/flower_experiment")
    if os.path.exists(save_path + "/weights") is False:
        os.makedirs(save_path + "/weights")

    img_size = 224
    data_transform = {
        "train": transforms.Compose([transforms.RandomResizedCrop(img_size),
                                     transforms.RandomHorizontalFlip(),
                                     transforms.ToTensor(),
                                     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]),
        "val": transforms.Compose([transforms.Resize(int(img_size * 1.143)),
                                   transforms.CenterCrop(img_size),
                                   transforms.ToTensor(),
                                   transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])}

    train_data_set = datasets.ImageFolder(root=os.path.join(args.data_path, "train"), transform=data_transform["train"])
    val_data_set = datasets.ImageFolder(root=os.path.join(args.data_path, "val"), transform=data_transform["val"])

    classes_list = train_data_set.class_to_idx
    cla_dict = dict((val, key) for key, val in classes_list.items())
    json_str = json.dumps(cla_dict, indent=4)
    with open('class_indices.json', 'w') as json_file:
        json_file.write(json_str)

    batch_size = args.batch_size
    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])
    print('Using {} dataloader workers every process'.format(nw))
    train_loader = torch.utils.data.DataLoader(train_data_set, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=nw)
    val_loader = torch.utils.data.DataLoader(val_data_set, batch_size=batch_size, shuffle=False, pin_memory=True, num_workers=nw)

    model = create_model(num_classes=args.num_classes).to(device)

    init_img = torch.zeros((1, 3, 224, 224), device=device)
    tb_writer.add_graph(model, init_img)

    if args.weights != "":
        assert os.path.exists(args.weights), "weights file: '{}' not exist.".format(args.weights)
        weights_dict = torch.load(args.weights, map_location=device)
        for k in list(weights_dict.keys()):
            if "head" in k:
                del weights_dict[k]
        print(model.load_state_dict(weights_dict, strict=False))

    if args.freeze_layers:
        for name, para in model.named_parameters():
            if "head" not in name:
                para.requires_grad_(False)
            else:
                print("training {}".format(name))

    pg = get_params_groups(model, weight_decay=args.wd)
    optimizer = optim.AdamW(pg, lr=args.lr, weight_decay=args.wd)
    lr_scheduler = create_lr_scheduler(optimizer, len(train_loader), args.epochs, warmup=True, warmup_epochs=1)

    best_acc = 0.0

    # 初始化训练和验证损失、准确度的列表
    train_losses = []
    val_losses = []
    train_accuracies = []
    val_accuracies = []

    for epoch in range(args.epochs):
        time_start = time.time()
        # train
        train_loss, train_acc = train_one_epoch(model=model,
                                                optimizer=optimizer,
                                                data_loader=train_loader,
                                                device=device,
                                                epoch=epoch,
                                                lr_scheduler=lr_scheduler)

        # validate
        val_loss, val_acc = evaluate(model=model,
                                     data_loader=val_loader,
                                     device=device,
                                     epoch=epoch)
        time_end = time.time()

        f.write("[epoch {}] train_loss: {:.3f},train_acc:{:.3f},val_loss:{:.3f},val_acc:{:.3f},Spend_time:{:.3f}S"
                .format(epoch + 1, train_loss, train_acc, val_loss, val_acc, time_end - time_start))
        f.flush()

        # 调用记录训练和验证损失的函数
        log_training_results(tb_writer, epoch, train_loss, train_acc, val_loss, val_acc, optimizer.param_groups[0]["lr"],
                             train_losses=train_losses, val_losses=val_losses,
                             train_accuracies=train_accuracies, val_accuracies=val_accuracies)

        log_validation_results(tb_writer, model, val_loader, epoch, device)

        if val_acc > best_acc:
            best_acc = val_acc
            f.write(',save best model')
            torch.save(model.state_dict(), save_path + "/weights/bestmodel.pth")
        f.write('\n')

    f.close()



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_classes', type=int, default=5)
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--lr', type=float, default=5e-4)
    parser.add_argument('--wd', type=float, default=5e-2)
    parser.add_argument('--data-path', type=str,
                        default=r"E:/Code/ConvNeXt-main/ConvNeXt/flower_data/")
    parser.add_argument('--weights', type=str,
                        default=r"E:/Code/ConvNeXt-main/convnext_tiny_1k_224_ema2.pth",
                        # default="",
                        help='initial weights path')
    parser.add_argument('--freeze-layers', type=bool, default=False)
    parser.add_argument('--device', default='cuda:0', help='device id (i.e. 0 or 0,1 or cpu)')

    opt = parser.parse_args()

    main(opt)
