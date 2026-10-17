"""
train.py
模型训练流程：
1. 固定随机种子（保证每次跑出来的结果一样）
2. 调 data_process 拿到 DataLoader 和预处理参数
3. 建模型、选损失函数和优化器
4. 多轮训练：一批一批喂数据 -> 算 loss -> 反向传播 -> 更新参数
5. 每一轮都在训练集和测试集上算 loss 和准确率，记下来
6. 训练完画两张图，存到 docs/
7. 保存模型权重和预处理参数，存到 output/
"""

import os
import pickle

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

from data_process import build_dataloaders, BASE_DIR
from model import TitanicMLP

# -------------------------- 超参数 --------------------------
SEED = 42          # 随机种子，固定之后结果可复现
EPOCHS = 200       # 训练轮数：每一轮会把训练集整个过一遍
LR = 1e-3          # 学习率：每次参数更新迈多大步子
BATCH_SIZE = 32    # 每批喂给模型 32 条数据
WEIGHT_DECAY = 1e-4  # L2 正则化强度：给过大的权重加惩罚，用来缓解过拟合

# 输出目录（用 __file__ 推导，避免运行时目录不同导致文件存错地方）
DOCS_DIR = os.path.join(BASE_DIR, "docs")
OUT_DIR = os.path.join(BASE_DIR, "output")


# -------------------------- 评估函数 --------------------------
def evaluate(model, data_loader, criterion):
    """
    在给定数据集上跑一遍，返回平均 loss 和准确率。
    torch.no_grad() 表示不算梯度，评估阶段不需要更新参数，能省内存和计算。
    """
    model.eval()                      # 切到评估模式
    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for xb, yb in data_loader:
            pred = model(xb)                       # 得到 0~1 的概率
            loss = criterion(pred, yb)
            total_loss += loss.item() * xb.size(0)  # 乘 batch 大小，方便后面算加权平均

            # 概率 > 0.5 判为幸存(1)，否则遇难(0)
            pred_label = (pred > 0.5).float()
            correct += (pred_label == yb).sum().item()
            total += yb.size(0)

    avg_loss = total_loss / total
    acc = correct / total
    return avg_loss, acc


# -------------------------- 主训练流程 --------------------------
def run_training():
    # 1. 固定随机种子（题目 3.3 可复现性要求）
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)

    # 2. 拿数据
    print("[1/6] 读取并预处理数据...")
    train_loader, test_loader, params = build_dataloaders(
        csv_path=os.path.join(BASE_DIR, "data", "train.csv"),
        batch_size=BATCH_SIZE,
        test_ratio=0.2,
        seed=SEED,
    )
    input_dim = len(params["feature_cols"])
    print(f"      特征维度: {input_dim}，训练批次数: {len(train_loader)}，测试批次数: {len(test_loader)}")

    # 3. 建模型、损失函数、优化器
    print("[2/6] 初始化模型...")
    model = TitanicMLP(input_dim=input_dim)
    criterion = nn.BCELoss()                    # 二分类交叉熵，专门用来比较"概率 vs 0/1标签"
    optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    # 4. 记录每轮的结果，后面画图要用
    train_loss_list, test_loss_list = [], []
    train_acc_list, test_acc_list = [], []

    print(f"[3/6] 开始训练，共 {EPOCHS} 轮...")
    for epoch in range(1, EPOCHS + 1):
        # --- 训练阶段 ---
        model.train()                            # 切回训练模式
        for xb, yb in train_loader:
            pred = model(xb)
            loss = criterion(pred, yb)

            optimizer.zero_grad()                # 清空上一轮残留的梯度
            loss.backward()                      # 反向传播，算出每个参数的梯度
            optimizer.step()                     # 按梯度更新参数

        # --- 每轮结束评估一次 ---
        train_loss, train_acc = evaluate(model, train_loader, criterion)
        test_loss, test_acc = evaluate(model, test_loader, criterion)

        train_loss_list.append(train_loss)
        test_loss_list.append(test_loss)
        train_acc_list.append(train_acc)
        test_acc_list.append(test_acc)

        # 每 10 轮打印一次，不然 200 行太刷屏
        if epoch == 1 or epoch % 10 == 0:
            print(f"      epoch {epoch:3d} | train loss {train_loss:.4f} | test loss {test_loss:.4f} | "
                  f"train acc {train_acc:.4f} | test acc {test_acc:.4f}")

    final_test_acc = test_acc_list[-1]
    print(f"\n[4/6] 训练完成，最终测试集准确率: {final_test_acc:.4f} ({final_test_acc*100:.2f}%)")

    # 5. 画图
    print("[5/6] 生成可视化图表...")
    os.makedirs(DOCS_DIR, exist_ok=True)
    epochs_axis = range(1, EPOCHS + 1)

    # 图1：损失趋势
    plt.figure(figsize=(8, 5))
    plt.plot(epochs_axis, train_loss_list, label="Train Loss", color="#1f77b4")
    plt.plot(epochs_axis, test_loss_list, label="Test Loss", color="#ff7f0e")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Model Loss Trend")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.savefig(os.path.join(DOCS_DIR, "loss_curve.png"), dpi=200, bbox_inches="tight")
    plt.close()

    # 图2：训练集和测试集准确率（同一张图两条线，方便看有没有过拟合）
    plt.figure(figsize=(8, 5))
    plt.plot(epochs_axis, train_acc_list, label="Train Accuracy", color="#2ca02c")
    plt.plot(epochs_axis, test_acc_list, label="Test Accuracy", color="#d62728")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Train & Test Accuracy")
    plt.legend()
    plt.ylim(0, 1)
    plt.grid(alpha=0.3)
    plt.savefig(os.path.join(DOCS_DIR, "accuracy_curve.png"), dpi=200, bbox_inches="tight")
    plt.close()
    print("      -> loss_curve.png / accuracy_curve.png 已存到 docs/")

    # 6. 保存模型和预处理参数
    print("[6/6] 保存模型与预处理参数...")
    os.makedirs(OUT_DIR, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(OUT_DIR, "titanic_model.pth"))
    with open(os.path.join(OUT_DIR, "preprocess_params.pkl"), "wb") as f:
        pickle.dump(params, f)
    print("      -> titanic_model.pth / preprocess_params.pkl 已存到 output/")

    history = {
        "train_loss": train_loss_list,
        "test_loss": test_loss_list,
        "train_acc": train_acc_list,
        "test_acc": test_acc_list,
        "final_test_acc": final_test_acc,
    }
    return model, params, history


if __name__ == "__main__":
    run_training()
