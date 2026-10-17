"""
data_process.py
负责：读数据 -> 清洗 -> 特征工程 -> 划分训练/测试集 -> 封装成 DataLoader

核心原则（题目 3.2 要求）：
所有"预处理参数"只能在训练集上算出来，测试集和新乘客都必须复用同一套参数，
否则就相当于让模型偷看了测试集的信息，评估出来的准确率是不真实的。
"""

import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

# 项目根目录 = src 的上级目录。用 __file__ 推导的好处是：
# 不管你在哪个目录下敲 python 命令，路径都不会错。
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "data", "train.csv")

# 这几个字段对预测没什么用：PassengerId 只是编号，Name/Ticket 是字符串，
# Cabin 缺失率高达 77% 补了也不准，所以直接丢掉。
DROP_COLS = ["PassengerId", "Name", "Ticket", "Cabin"]

# 数值特征（会被标准化）和类别特征（会被 one-hot 展开）
NUM_COLS = ["Age", "Fare", "SibSp", "Parch", "FamilySize"]
CAT_COLS = ["Pclass", "Sex", "Embarked"]


# ------------------------------------------------------------
# 1. 读数据 + 基础清洗
# ------------------------------------------------------------
def read_raw(csv_path=DATA_PATH):
    """读原始 csv"""
    return pd.read_csv(csv_path)


def basic_clean(df):
    """
    丢掉无用列 + 构造家庭规模特征。
    这一步不需要任何统计量（没有算均值之类的），所以可以放心地对整份数据先做。
    """
    df = df.copy()
    df = df.drop(columns=[c for c in DROP_COLS if c in df.columns])

    # 家庭人数 = 兄弟姐妹配偶数 + 父母子女数 + 自己
    # 直觉上：一个人逃生的和一家老小一起的，生存率不一样，所以这个特征有用
    df["FamilySize"] = df["SibSp"] + df["Parch"] + 1
    return df


# ------------------------------------------------------------
# 2. 在训练集上算预处理参数
# ------------------------------------------------------------
def fit_params(train_df):
    """
    只传训练集进来，算出后面要用到的全部参数，塞进一个字典返回。
    这些参数会被保存下来，预测新乘客时再读出来用。
    """
    params = {}
    df = train_df.copy()

    # --- 2.1 缺失值填充值 ---
    # Age 和 Fare 是数值，用中位数填。为什么不用均值？
    # 因为票价/年龄里有少数特别大的数，会把均值拉偏，中位数更稳。
    params["fill_age"] = float(df["Age"].median())
    params["fill_fare"] = float(df["Fare"].median())
    # Embarked 是类别，只能填出现次数最多的那个（众数）
    params["fill_embarked"] = str(df["Embarked"].mode()[0])

    # 先把训练集自己的缺失填上，再算标准化参数，不然 mean/std 会被 NaN 影响
    df["Age"] = df["Age"].fillna(params["fill_age"])
    df["Fare"] = df["Fare"].fillna(params["fill_fare"])
    df["Embarked"] = df["Embarked"].fillna(params["fill_embarked"])

    # --- 2.2 数值特征的均值和标准差（用于标准化）---
    # 神经网络对输入尺度很敏感：Fare 能到几百，Age 只有几十，
    # 不归一化会让训练很难收敛，所以统一压缩到"均值0、标准差1"附近。
    params["num_cols"] = NUM_COLS
    params["num_mean"] = {c: float(df[c].mean()) for c in NUM_COLS}
    params["num_std"] = {c: float(df[c].std()) for c in NUM_COLS}

    # --- 2.3 类别特征有哪些取值（决定 one-hot 展开成几列）---
    # 必须用训练集见过的取值，预测时万一来了新类别也能保证列数一致
    params["cat_cols"] = CAT_COLS
    params["cat_values"] = {}
    for c in CAT_COLS:
        params["cat_values"][c] = sorted(df[c].dropna().unique().tolist())

    # --- 2.4 最终喂给模型的特征列顺序（顺序必须固定，否则训练和预测会错位）---
    params["feature_cols"] = NUM_COLS + [
        f"{c}_{v}" for c in CAT_COLS for v in params["cat_values"][c]
    ]
    return params


# ------------------------------------------------------------
# 3. 用参数处理数据（训练集/测试集/新乘客都走这个函数）
# ------------------------------------------------------------
def apply_preprocess(df, params):
    """
    把任意一份原始表格（训练、测试、或者一个刚输入的新乘客）
    转成模型能吃的 float32 矩阵。全程只用 params 里存好的参数，不重新统计。
    """
    df = df.copy()

    # 新乘客是单条字典转成的 DataFrame，还没算过 FamilySize，这里补上
    if "FamilySize" not in df.columns:
        df["SibSp"] = df["SibSp"].fillna(0)
        df["Parch"] = df["Parch"].fillna(0)
        df["FamilySize"] = df["SibSp"] + df["Parch"] + 1

    # 1) 缺失值：用训练集算好的值填
    df["Age"] = df["Age"].fillna(params["fill_age"])
    df["Fare"] = df["Fare"].fillna(params["fill_fare"])
    df["Embarked"] = df["Embarked"].fillna(params["fill_embarked"])

    # 2) 数值特征标准化
    for c in params["num_cols"]:
        std = params["num_std"][c]
        if std < 1e-8:      # 防止某个特征全是同一个值导致除以 0
            std = 1.0
        df[c] = (df[c] - params["num_mean"][c]) / std

    # 3) 类别特征 one-hot
    # 比如 Sex 有 female/male 两档，就展开成 Sex_female、Sex_male 两列，
    # 对应的那一列填 1，另一列填 0
    for c in params["cat_cols"]:
        for v in params["cat_values"][c]:
            df[f"{c}_{v}"] = (df[c] == v).astype(float)

    # 按固定的列顺序取出特征矩阵
    X = df[params["feature_cols"]].values.astype(np.float32)
    return X


# ------------------------------------------------------------
# 4. 包装成 PyTorch 的 Dataset / DataLoader
# ------------------------------------------------------------
class TitanicDataset(Dataset):
    """把 numpy 数组包装成 PyTorch 的 Dataset，DataLoader 才能一批批取数据"""

    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32).view(-1, 1)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def build_dataloaders(csv_path=DATA_PATH, batch_size=32, test_ratio=0.2, seed=42):
    """
    完整流程：读数据 -> 清洗 -> 划分 -> 算参数 -> 转换 -> 返回两个 DataLoader

    注意顺序：必须先划分，再用训练集算参数。
    如果先拿全量数据算均值再划分，测试集的信息就泄露进预处理了。
    """
    np.random.seed(seed)

    raw = read_raw(csv_path)
    df = basic_clean(raw)

    # --- 划分：随机打乱下标，前 20% 当测试集，后 80% 当训练集 ---
    n = len(df)
    idx = np.random.permutation(n)
    test_size = int(n * test_ratio)
    test_idx = idx[:test_size]
    train_idx = idx[test_size:]

    train_df = df.iloc[train_idx].reset_index(drop=True)
    test_df = df.iloc[test_idx].reset_index(drop=True)

    # --- 参数只在训练集上算 ---
    params = fit_params(train_df)

    # --- 训练集和测试集用同一套参数转换 ---
    X_train = apply_preprocess(train_df, params)
    X_test = apply_preprocess(test_df, params)

    y_train = train_df["Survived"].values.astype(np.float32)
    y_test = test_df["Survived"].values.astype(np.float32)

    train_loader = DataLoader(
        TitanicDataset(X_train, y_train), batch_size=batch_size, shuffle=True
    )
    test_loader = DataLoader(
        TitanicDataset(X_test, y_test), batch_size=batch_size, shuffle=False
    )

    return train_loader, test_loader, params


if __name__ == "__main__":
    # 单独运行这个文件可以自检：看看特征长什么样、一个 batch 的形状对不对
    train_loader, test_loader, params = build_dataloaders(DATA_PATH)
    print("特征列:", params["feature_cols"])
    print("特征维度:", len(params["feature_cols"]))
    print("训练批次数:", len(train_loader), " 测试批次数:", len(test_loader))
    xb, yb = next(iter(train_loader))
    print("一个 batch 输入 shape:", xb.shape, " 标签 shape:", yb.shape)
