import pandas as pd
import os
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torch.nn as nn
import torchvision.models as models
import torch
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from torchvision.models import ConvNeXt_Base_Weights
import time  # 引入time模块


# 自定义数据集
class PetAgeDataset(Dataset):
    def __init__(self, image_folder, annotation_file, transform=None):
        self.image_folder = image_folder
        self.annotations = pd.read_csv(annotation_file, delimiter='\t', header=None, names=['image_name', 'age_month'])
        self.transform = transform

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        image_name = self.annotations.iloc[idx, 0]
        image_path = os.path.join(self.image_folder, image_name)
        image = Image.open(image_path).convert('RGB')
        age = int(self.annotations.iloc[idx, 1])

        if self.transform:
            image = self.transform(image)

        return image, age



class SEAttention(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SEAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


# 构建ConvNeXt回归模型
class ConvNeXtAgeModel(nn.Module):
    def __init__(self):
        super(ConvNeXtAgeModel, self).__init__()
        # 使用预训练的ConvNeXt
        base_model = models.convnext_base(weights=ConvNeXt_Base_Weights.DEFAULT)
        children = list(base_model.children())
        # 冻结前若干层，这里示例冻结前3层，可根据实际调整
        for layer in children[:3]:
            for param in layer.parameters():
                param.requires_grad = False
        self.convnext = nn.Sequential(*list(base_model.children())[:-1])  # 去掉最后分类层
        self.attention = SEAttention(1024)  # 针对ConvNeXt输出特征维度为1024来设置
        # self.fc = nn.Linear(1024, 1)  # ConvNeXt输出特征维度为1024
        self.fc1 = nn.Linear(1024, 512)  # 新增一层
        self.bn = nn.BatchNorm1d(512)  # 批归一化层
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(512, 1)

    def forward(self, x):
        x = self.convnext(x)
        x = self.attention(x)  # 添加注意力机制
        x = x.mean([2, 3])  # 全局平均池化
        # x = self.fc(x)
        x = self.fc1(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x


# 训练函数
def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, num_epochs=10):
    # scaler = GradScaler()  # 混合精度训练
    scaler = torch.amp.GradScaler('cuda')
    best_model_wts = model.state_dict()
    best_mae = float('inf')
    train_losses, val_losses = [], []
    ii = 1
    for epoch in range(num_epochs):
        start_time = time.time()  # 记录每个epoch开始时间
        model.train()
        running_loss = 0.0

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.float().to(device)
            optimizer.zero_grad()

            # with autocast():  # 混合精度训练
            with torch.amp.autocast('cuda'):  # 混合精度训练
                outputs = model(inputs).squeeze()
                loss = criterion(outputs, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item() * inputs.size(0)

        epoch_train_loss = running_loss / len(train_loader.dataset)
        train_losses.append(epoch_train_loss)

        # 验证阶段
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.float().to(device)
                outputs = model(inputs).squeeze()
                loss = criterion(outputs, labels)
                val_loss += loss.item() * inputs.size(0)

        epoch_val_loss = val_loss / len(val_loader.dataset)
        val_losses.append(epoch_val_loss)
        scheduler.step()

        end_time = time.time()  # 记录每个epoch结束时间
        epoch_time = end_time - start_time  # 计算每个epoch花费的时间
        print(f"Epoch {epoch + 1}/{num_epochs}, Train Loss: {epoch_train_loss:.4f}, Val Loss: {epoch_val_loss:.4f}, Epoch Time: {epoch_time:.2f} seconds")

        if epoch_val_loss < best_mae:
            best_mae = epoch_val_loss
            best_model_wts = model.state_dict()

    model.load_state_dict(best_model_wts)
    return model, train_losses, val_losses


# 评估模型
def evaluate_model(model, val_loader):
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs).squeeze()
            all_preds.append(outputs.cpu().numpy())
            all_labels.append(labels.cpu().numpy())
    all_preds = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)
    mae = np.mean(np.abs(all_preds - all_labels))
    print(f"Mean Absolute Error (MAE): {mae:.4f}")
    return all_preds, all_labels

def save_predictions(pred_ages, image_name, output_file='pred_result.txt'):
    with open(output_file, 'w') as f:
        for idx, age in enumerate(pred_ages):
            f.write(f'{image_name.iloc[idx, 0]}\t{age}\n')

# 主函数
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # 数据增强和预处理
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        transforms.GaussianBlur(kernel_size=(3, 3), sigma=(0.1, 2.0))  # 示例参数，可调整
        # 后续操作
    ])

    # 数据集
    train_dataset = PetAgeDataset(image_folder='./data/trainset/trainset',
                                  annotation_file='./data/annotations/annotations/train.txt',
                                  transform=transform)
    val_dataset = PetAgeDataset(image_folder='./data/valset/valset',
                                annotation_file='./data/annotations/annotations/val.txt',
                                transform=transform)

    # DataLoader
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=0, pin_memory=True)

    # 模型、损失函数和优化器
    model = ConvNeXtAgeModel().to(device)
    criterion = nn.SmoothL1Loss()  # 稳健损失函数
    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)

 
    # 训练模型
    model, train_losses, val_losses = train_model(model, train_loader, val_loader, criterion, optimizer, scheduler,
                                                  num_epochs=200)

    # 评估模型
    pred_ages, true_ages = evaluate_model(model, val_loader)

    image_name = pd.read_csv('E:/Code/data/petage/annotations/annotations/val.txt', delimiter='\t', header=None,
                             names=['image_name', 'age_month'])

    # 保存预测结果
    save_predictions(pred_ages, image_name)

    # 可视化损失曲线
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.show()
