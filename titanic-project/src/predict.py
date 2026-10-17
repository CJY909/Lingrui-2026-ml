"""
predict.py
新乘客预测：输入一名乘客的个人信息，输出他是否幸存（0 = 遇难，1 = 幸存）

关键：新乘客必须复用训练时保存下来的那套预处理参数，
不能自己再统计一遍，否则特征尺度跟训练时不一致，预测就没意义了。
预测结果完全由训练好的 PyTorch 模型算出来，没有写任何人工规则。
"""

import os
import pickle

import pandas as pd
import torch

from data_process import apply_preprocess, BASE_DIR
from model import TitanicMLP

MODEL_PATH = os.path.join(BASE_DIR, "output", "titanic_model.pth")
PARAMS_PATH = os.path.join(BASE_DIR, "output", "preprocess_params.pkl")


def load_model_and_params(model_path=MODEL_PATH, params_path=PARAMS_PATH):
    """
    把训练时保存下来的模型权重和预处理参数重新读出来。
    注意 input_dim 要和训练时一致，所以从 params 里取特征列数量。
    """
    if not os.path.exists(model_path) or not os.path.exists(params_path):
        raise FileNotFoundError(
            "没找到模型文件，请先运行 train.py 或 main.py 完成训练。"
        )

    with open(params_path, "rb") as f:
        params = pickle.load(f)

    model = TitanicMLP(input_dim=len(params["feature_cols"]))
    model.load_state_dict(torch.load(model_path))
    model.eval()                 # 评估模式，预测时必须加
    return model, params


def predict_one(passenger: dict, model, params):
    """
    预测单个乘客。
    passenger 是一个字典，至少要包含这些键：
        Pclass   舱位（1 / 2 / 3）
        Sex      性别（"male" / "female"）
        Age      年龄（数字，不知道就填 None）
        SibSp    同行兄弟姐妹/配偶数
        Parch    同行父母/子女数
        Fare     票价
        Embarked 登船港口（"C" / "Q" / "S"）

    返回：(0 或 1, 生还概率)
    """
    df = pd.DataFrame([passenger])          # 单条数据也转成表格，复用同一套处理函数
    X = apply_preprocess(df, params)        # 用训练时的参数做转换

    x = torch.tensor(X, dtype=torch.float32)
    with torch.no_grad():                   # 预测不需要算梯度
        prob = model(x).item()              # 模型输出的 0~1 概率

    label = 1 if prob > 0.5 else 0          # 大于 0.5 判为幸存
    return label, prob


if __name__ == "__main__":
    # 几个例子，覆盖不同情况，看看模型学得合不合理
    examples = [
        {"Pclass": 1, "Sex": "female", "Age": 25, "SibSp": 0, "Parch": 0, "Fare": 100.0, "Embarked": "S"},
        {"Pclass": 3, "Sex": "male",   "Age": 30, "SibSp": 0, "Parch": 0, "Fare": 8.0,   "Embarked": "S"},
        {"Pclass": 2, "Sex": "female", "Age": 8,  "SibSp": 1, "Parch": 2, "Fare": 30.0,  "Embarked": "C"},
        {"Pclass": 3, "Sex": "male",   "Age": 45, "SibSp": 4, "Parch": 2, "Fare": 20.0,  "Embarked": "Q"},
    ]

    model, params = load_model_and_params()
    print("新乘客预测结果：")
    print("-" * 62)
    for p in examples:
        label, prob = predict_one(p, model, params)
        result = "幸存" if label == 1 else "遇难"
        print(f"舱位{p['Pclass']} | {p['Sex']:>6} | {p['Age']:>2}岁 | "
              f"票价{p['Fare']:>6.1f} -> 预测 {label}({result})，生还概率 {prob:.3f}")
