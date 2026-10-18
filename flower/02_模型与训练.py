# -*- coding: utf-8 -*-
# 用预处理好的 flower_data 训练花卉分类模型
# 网络结构完全照着作业图里的来：C1~C5 五个卷积层 + FC6/FC7/FC8 三个全连接层
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib
matplotlib.use('Agg')  # 不弹窗口，直接存图
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['SimHei']  # 让图里能显示中文
plt.rcParams['axes.unicode_minus'] = False

BASE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(BASE, 'flower_data')

# 服务器/无界面环境下画图用不了默认后端，强制换成 Agg，只存文件不弹窗

# ---------- 数据增强 ----------
# 训练集做随机增强，相当于"扩充"了样本的多样性
# 尺寸统一到 224x224，正好是网络结构图里输入的大小
train_tf = transforms.Compose([
    transforms.Resize(256),                 # 先放大到短边256
    transforms.RandomCrop(224),             # 再随机裁一块224的，每次位置不一样
    transforms.RandomHorizontalFlip(),      # 随机左右翻转
    transforms.RandomRotation(15),          # 随机旋转±15度
    transforms.ColorJitter(0.2, 0.2, 0.2),  # 随机调亮度/对比度/饱和度
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # 用ImageNet的均值方差
])
# 验证集和测试集不能做随机增强，只要中心裁剪，保证评估结果稳定
eval_tf = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])


# ---------- 网络结构（对应作业图） ----------
# 图里各层：C1卷积核11*11步长4输出96 -> 池化 -> C2卷积核5*5输出256 -> 池化
#          -> C3/C4/C5卷积核3*3，输出384/384/256 -> 池化 -> FC6 4096 -> FC7 4096 -> FC8
class Mymodel(nn.Module):
    def __init__(self, num_classes=5):
        super(Mymodel, self).__init__()
        # C1：输入3通道，卷积核11x11，步长4。原图224，padding=2算出来正好是55，跟图上一致
        self.conv1 = nn.Conv2d(3, 96, kernel_size=11, stride=4, padding=2)
        # C2：输入96通道，卷积核5x5，输出256
        self.conv2 = nn.Conv2d(96, 256, kernel_size=5, padding=2)
        # C3、C4：3x3卷积，输出都是384
        self.conv3 = nn.Conv2d(256, 384, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(384, 384, kernel_size=3, padding=1)
        # C5：3x3卷积，输出256
        self.conv5 = nn.Conv2d(384, 256, kernel_size=3, padding=1)
        # 图里三处 Max pooling，都是同一个，可以共用
        self.pool = nn.MaxPool2d(kernel_size=3, stride=2)
        # FC8 的输出类别数：图上写的是18，但我们数据集只有5种花，所以改成5
        self.fc = nn.Sequential(
            nn.Linear(256 * 6 * 6, 4096),   # FC6，256*6*6=9216 是C5池化后展平的长度
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(4096, 4096),          # FC7
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(4096, num_classes)    # FC8
        )

    def forward(self, x):
        # 输入 224x224x3
        x = self.pool(torch.relu(self.conv1(x)))   # -> 55x55x96 -> 27x27x96
        x = self.pool(torch.relu(self.conv2(x)))   # -> 27x27x256 -> 13x13x256
        x = torch.relu(self.conv3(x))              # -> 13x13x384
        x = torch.relu(self.conv4(x))              # -> 13x13x384
        x = self.pool(torch.relu(self.conv5(x)))   # -> 13x13x256 -> 6x6x256
        x = x.view(x.size(0), -1)                  # 展平成一维向量
        x = self.fc(x)
        return x


# ---------- 训练和验证各跑一轮 ----------
def run_epoch(model, loader, criterion, optimizer=None):
    training = optimizer is not None
    model.train() if training else model.eval()
    total_loss, correct, total = 0.0, 0, 0
    with torch.set_grad_enabled(training):
        for imgs, labels in loader:
            out = model(imgs)
            loss = criterion(out, labels)
            if training:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            total_loss += loss.item() * imgs.size(0)
            correct += (out.argmax(1) == labels).sum().item()
            total += imgs.size(0)
    return total_loss / total, correct / total


if __name__ == '__main__':
    train_set = datasets.ImageFolder(os.path.join(DATA, 'train'), train_tf)
    val_set = datasets.ImageFolder(os.path.join(DATA, 'val'), eval_tf)
    test_set = datasets.ImageFolder(os.path.join(DATA, 'test'), eval_tf)
    print('类别顺序:', train_set.classes)

    # batch_size=32 一次喂32张图，CPU上这个大小比较稳
    train_loader = DataLoader(train_set, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=32)
    test_loader = DataLoader(test_set, batch_size=32)

    model = Mymodel(num_classes=5)
    criterion = nn.CrossEntropyLoss()                       # 多分类就用交叉熵
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)  # Adam调参省心，学习率1e-4起步

    EPOCHS = 10
    train_acc_list, val_acc_list, train_loss_list, val_loss_list = [], [], [], []

    for epoch in range(EPOCHS):
        tr_loss, tr_acc = run_epoch(model, train_loader, criterion, optimizer)
        va_loss, va_acc = run_epoch(model, val_loader, criterion)
        train_acc_list.append(tr_acc)
        val_acc_list.append(va_acc)
        train_loss_list.append(tr_loss)
        val_loss_list.append(va_loss)
        print('epoch %2d | 训练损失 %.4f 训练准确率 %.2f%% | 验证损失 %.4f 验证准确率 %.2f%%'
              % (epoch + 1, tr_loss, tr_acc * 100, va_loss, va_acc * 100))

    # 最后用没参与训练的测试集测一次
    te_loss, te_acc = run_epoch(model, test_loader, criterion)
    print('测试集结果 | 损失 %.4f 准确率 %.2f%%' % (te_loss, te_acc * 100))

    # ---------- 画准确率/损失曲线 ----------
    x = range(1, EPOCHS + 1)
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.plot(x, train_acc_list, 'o-', label='训练准确率')
    plt.plot(x, val_acc_list, 's-', label='验证准确率')
    plt.xlabel('epoch')
    plt.ylabel('准确率')
    plt.title('准确率随训练进度变化')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.subplot(1, 2, 2)
    plt.plot(x, train_loss_list, 'o-', label='训练损失')
    plt.plot(x, val_loss_list, 's-', label='验证损失')
    plt.xlabel('epoch')
    plt.ylabel('损失')
    plt.title('损失随训练进度变化')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(BASE, '训练曲线.png'), dpi=150)

    # 保存模型，以后想用直接加载，不用重新训练
    torch.save(model.state_dict(), os.path.join(BASE, 'flower_model.pth'))
    print('已保存 训练曲线.png 和 flower_model.pth')
