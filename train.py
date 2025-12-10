import warnings
# 忽视警告
warnings.filterwarnings('ignore')

import cv2
from PIL import Image
import numpy as np
import copy
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models
from torchvision.datasets import ImageFolder
import torchvision.transforms as T
from torch.utils.data import DataLoader
from torch_py.Utils import plot_image
from torch_py.MTCNN.detector import FaceDetector
from torch_py.MobileNetV1 import MobileNetV1
from torch_py.FaceRec import Recognition
# 1.加载数据并进行数据处理
def letterbox_image(image, size):
    """
    调整图片尺寸
    :param image: 用于训练的图片
    :param size: 需要调整到网络输入的图片尺寸
    :return: 返回经过调整的图片
    """
    new_image = cv2.resize(image, size, interpolation=cv2.INTER_AREA)
    return new_image

def processing_data(data_path, height=224, width=224, batch_size=32,
                    test_split=0.1):
    """
    数据处理部分
    :param data_path: 数据路径
    :param height:高度
    :param width: 宽度
    :param batch_size: 每次读取图片的数量
    :param test_split: 测试集划分比例
    :return: 
    """
    transforms = T.Compose([
        T.Resize((height, width)),
        T.RandomHorizontalFlip(0.1),  # 进行随机水平翻转
        T.RandomVerticalFlip(0.1),  # 进行随机竖直翻转
        T.ToTensor(),  # 转化为张量
        T.Normalize([0], [1]),  # 归一化
    ])

    dataset = ImageFolder(data_path, transform=transforms)
    # 划分数据集
    train_size = int((1-test_split)*len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])
    # 创建一个 DataLoader 对象
    train_data_loader = DataLoader(train_dataset, batch_size=batch_size,shuffle=True)
    valid_data_loader = DataLoader(test_dataset, batch_size=batch_size,shuffle=True)

    return train_data_loader, valid_data_loader
data_path = './datasets/5f680a696ec9b83bb0037081-momodel/data/image'
train_data_loader, valid_data_loader = processing_data(data_path=data_path, height=160, width=160, batch_size=32)

def show_tensor_img(img_tensor):
    img = img_tensor[0].data.numpy()
    img = np.swapaxes(img, 0, 2)
    img = np.swapaxes(img, 0, 1)
    img = np.array(img)
    plot_image(img)
# 2.如果有预训练模型，则加载预训练模型；如果没有则不需要加载

# 加载 MobileNet 的预训练模型权
device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")

train_data_loader, valid_data_loader = processing_data(data_path=data_path, height=160, width=160, batch_size=32)
modify_x, modify_y = torch.ones((32, 3, 160, 160)), torch.ones((32))

epochs = 50
NUM_CLASSES=2
try:
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
except AttributeError:
    # 兼容旧版本，使用 pretrained=True
    model = models.resnet18(pretrained=True)
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, NUM_CLASSES)
model = model.to(device)
optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)  # 优化器
print('加载完成...')
# 3.创建模型和训练模型，训练模型时尽量将模型保存在 results 文件夹
# 学习率下降的方式，acc三次不下降就下降学习率继续训练，衰减学习率
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                 'min', 
                                                 factor=0.5,
                                                 patience=2)
# 损失函数
criterion = nn.CrossEntropyLoss()  

best_loss = 1e9
best_model_weights = copy.deepcopy(model.state_dict())
# 确保在循环外部定义了 scheduler 和 best_loss/best_model_weights

train_loss_list = []  # 存储平均训练损失
valid_loss_list = []

for epoch in range(epochs):
    # 初始化本epoch的运行损失
    running_train_loss = 0.0 
    model.train()

    # --- 训练阶段 ---
    train_pbar = tqdm(enumerate(train_data_loader, 1), total=len(train_data_loader), desc=f'Epoch {epoch+1}/{epochs} [Train]')
    for batch_idx, (x, y) in train_pbar:
        x = x.to(device)
        y = y.to(device)
        pred_y = model(x)

        loss = criterion(pred_y, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_train_loss += loss.item()
        # 更新进度条，显示当前批次的损失
        train_pbar.set_postfix(loss=f'{loss.item():.4f}', lr=f'{optimizer.param_groups[0]["lr"]:.1e}')
        
    # 计算并记录平均训练损失
    avg_train_loss = running_train_loss / len(train_data_loader)
    train_loss_list.append(avg_train_loss)
    # --- 验证阶段 ---
    model.eval()
    running_val_loss = 0.0
    with torch.no_grad():
        val_pbar = tqdm(valid_data_loader, desc=f'Epoch {epoch+1}/{epochs} [Validation]')
        for x, y in val_pbar:
            x = x.to(device)
            y = y.to(device)

            pred_y = model(x)
            loss = criterion(pred_y, y)
            running_val_loss += loss.item()
            # 更新进度条显示当前批次的损失
            val_pbar.set_postfix(loss=f'{loss.item():.4f}')

    # 计算并记录平均验证损失
    avg_val_loss = running_val_loss / len(valid_data_loader)
    valid_loss_list.append(avg_val_loss)
    # --- 模型保存与学习率调度 (修正后的关键步骤) ---

    # 1. 调度器步进：根据平均验证损失调整学习率
    if scheduler is not None:
        scheduler.step(avg_val_loss)

    # 2. 模型保存：仅根据平均验证损失判断是否保存最佳模型
    if avg_val_loss < best_loss:
        print(f'*** Validation Loss improved from {best_loss:.4f} to {avg_val_loss:.4f}. Saving model... ***')
        best_loss = avg_val_loss
        best_model_weights = copy.deepcopy(model.state_dict()) # <--- 现在保存是基于泛化能力的

    # 3. 打印 Epoch 总结
    print(f'Step: {epoch + 1}/{epochs} || Avg Train Loss: {avg_train_loss:.4f} || Avg Val Loss: {avg_val_loss:.4f} || Current LR: {optimizer.param_groups[0]["lr"]:.1e}')
torch.save(model.state_dict(), './results/temp.pth')
print('Finish Training.')
plt.figure(figsize=(10, 6))
plt.plot(train_loss_list, label='Training Loss')
plt.plot(valid_loss_list, label='Validation Loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)
plt.savefig('./results/loss_curve.png') # 也可以保存图片
plt.show()
# 4.评估模型，将自己认为最佳模型保存在 result 文件夹，其余模型备份在项目中其它文件夹，方便您加快测试通过。

model_path = 'results/temp.pth'
# ---------------------------------------------------------------------------

def predict(img):
    """
    加载模型和模型预测
    :param img: cv2.imread 图像
    :return: 预测的图片中的总人数、其中佩戴口罩的人数
    """
    # -------------------------- 实现模型预测部分的代码 ---------------------------
    # 将 cv2.imread 图像转化为 PIL.Image 图像，用来兼容测试输入的 cv2 读取的图像（勿删！！！）
    # cv2.imread 读取图像的类型是 numpy.ndarray
    # PIL.Image.open 读取图像的类型是 PIL.JpegImagePlugin.JpegImageFile
    if isinstance(img, np.ndarray):
        # 转化为 PIL.JpegImagePlugin.JpegImageFile 类型
        img = Image.fromarray(cv2.cvtColor(img,cv2.COLOR_BGR2RGB))
    
    recognize = Recognition(model_path)
    img, all_num, mask_num = recognize.mask_recognize(img)
    # -------------------------------------------------------------------------
    return all_num,mask_num