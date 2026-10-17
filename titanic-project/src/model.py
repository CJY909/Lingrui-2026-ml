"""
model.py
定义二分类神经网络模型（继承 torch.nn.Module）

结构：13 个输入特征 -> 16 -> 8 -> 1 个输出（Sigmoid 输出生还概率）

为什么用这么简单的三层全连接？
泰坦尼克号只有 891 条数据、13 个特征，属于很小的表格型数据集。
这种数据量下，堆很深的网络的收益很小，反而容易过拟合，
所以用一个浅一点的小网络就够了，重点是先把流程跑通。
"""

import torch
import torch.nn as nn


class TitanicMLP(nn.Module):
    """
    简单多层感知机（MLP）：
    输入 13 维特征，输出 1 维概率（0~1 之间的小数，表示生还概率）。
    """

    def __init__(self, input_dim: int = 13, hidden1: int = 16, hidden2: int = 8):
        super().__init__()                        # 必须先初始化父类，不然会报错
        self.fc1 = nn.Linear(input_dim, hidden1)  # 13 -> 16
        self.fc2 = nn.Linear(hidden1, hidden2)    # 16 -> 8
        self.fc3 = nn.Linear(hidden2, 1)          # 8  -> 1
        self.relu = nn.ReLU()                     # 激活函数，给网络引入非线性
        self.sigmoid = nn.Sigmoid()               # 把输出压到 0~1，当成概率用

    def forward(self, x):
        """
        前向计算（模型怎么把输入变成输出）。
        输入 x 形状: [batch_size, 13]
        输出形状:    [batch_size, 1]
        """
        x = self.relu(self.fc1(x))       # 第一层全连接 + ReLU
        x = self.relu(self.fc2(x))       # 第二层全连接 + ReLU
        x = self.sigmoid(self.fc3(x))    # 输出层 + Sigmoid，得到 0~1 的概率
        return x


if __name__ == "__main__":
    # ---- 自检：确认模型能正常前向计算，输入输出形状正确 ----
    model = TitanicMLP()
    print(model)                                   # 打印网络结构

    dummy = torch.randn(4, 13)                     # 造 4 条假数据，每条 13 个特征
    out = model(dummy)
    print("输入 shape:", dummy.shape)              # 期望 [4, 13]
    print("输出 shape:", out.shape)                # 期望 [4, 1]
    print("输出值范围:", round(out.min().item(), 4), "~", round(out.max().item(), 4))
