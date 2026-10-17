"""
main.py
一键运行入口：训练 -> 评估 -> 画图 -> 保存模型 -> 新乘客预测演示

直接运行这个文件就能把整个作业流程从头到尾跑一遍：
    python main.py
"""

from train import run_training
from predict import load_model_and_params, predict_one


def main():
    print("=" * 62)
    print("泰坦尼克号生还预测 —— PyTorch 全流程")
    print("=" * 62)

    # 第 1 步：训练模型（内部会画图、保存模型和预处理参数）
    model, params, history = run_training()

    # 第 2 步：用训练好的模型预测新乘客
    print("\n" + "=" * 62)
    print("新乘客预测演示")
    print("=" * 62)

    model, params = load_model_and_params()

    examples = [
        {"Pclass": 1, "Sex": "female", "Age": 25, "SibSp": 0, "Parch": 0, "Fare": 100.0, "Embarked": "S"},
        {"Pclass": 3, "Sex": "male",   "Age": 30, "SibSp": 0, "Parch": 0, "Fare": 8.0,   "Embarked": "S"},
        {"Pclass": 2, "Sex": "female", "Age": 8,  "SibSp": 1, "Parch": 2, "Fare": 30.0,  "Embarked": "C"},
        {"Pclass": 3, "Sex": "male",   "Age": 45, "SibSp": 4, "Parch": 2, "Fare": 20.0,  "Embarked": "Q"},
    ]

    for p in examples:
        label, prob = predict_one(p, model, params)
        result = "幸存" if label == 1 else "遇难"
        print(f"舱位{p['Pclass']} | {p['Sex']:>6} | {p['Age']:>2}岁 | "
              f"票价{p['Fare']:>6.1f} -> 预测 {label}({result})，生还概率 {prob:.3f}")

    print("\n全部流程完成！")
    print("  图表: docs/loss_curve.png、docs/accuracy_curve.png")
    print("  模型: output/titanic_model.pth、output/preprocess_params.pkl")


if __name__ == "__main__":
    main()
