import numpy as np
import os
import sys
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score
from base_classifiers import LogisticRegression, DecisionStump
from adaboost import AdaBoost

def load_data(data_path, target_path):
    """加载数据并确保标签格式一致"""
    # 使用更健壮的文件读取方式
    try:
        X = pd.read_csv(data_path).values
        y = pd.read_csv(target_path).values.flatten()
    except Exception as e:
        raise ValueError(f"数据加载失败: {str(e)}")
    
    # 确保标签为0/1格式（与输出一致）
    if not np.array_equal(np.unique(y), np.array([0, 1])):
        y = np.where(y > 0, 1, 0)  # 将非0值转换为1
    
    return X, y

def run_cross_validation(X, y, base_type, output_dir):
    """执行10折交叉验证并保存结果"""
    # 创建输出目录（如果不存在）
    os.makedirs(output_dir, exist_ok=True)
    
    # 配置交叉验证
    kf = KFold(n_splits=10, shuffle=True, random_state=42)
    
    # 确定基础学习器类型
    base_learner = LogisticRegression if base_type == 0 else DecisionStump
    display_name = 'Logistic Regression' if base_type == 0 else 'Decision Stump'
    num_classifiers_options = [1, 5, 10, 100]
    
    # 为每个分类器数量配置执行验证
    for num_classifiers in num_classifiers_options:
        accuracies = []
        print(f"\n使用基础学习器: {display_name}, 分类器数量: {num_classifiers}")
        
        # 执行10折交叉验证
        for fold, (train_idx, test_idx) in enumerate(kf.split(X), 1):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]
            
            # 训练和预测
            model = AdaBoost(base_learner, num_classifiers)
            model.fit(X_train, y_train)
            predictions = model.predict(X_test)
            
            # 保存预测结果（0/1标签）
            results = np.column_stack((test_idx + 1, predictions))
            filename = os.path.join(output_dir, f'base{num_classifiers}_fold{fold}.csv')
            np.savetxt(filename, results, fmt='%d,%d', header='Index,Prediction', comments='')
            
            # 计算准确率（确保y_test与predictions标签格式一致）
            accuracy = accuracy_score(y_test, predictions)
            accuracies.append(accuracy)
            print(f"Fold {fold} 准确率: {accuracy:.4f}")
        
        # 计算平均准确率
        avg_accuracy = np.mean(accuracies)
        print(f"{num_classifiers}个分类器的平均准确率: {avg_accuracy:.4f}\n")

def main():

    if len(sys.argv) != 4:
        print("用法: python main.py /path/to/data/data.csv /path/to/data/target.csv [0|1]")
        print("0: Logistic Regression, 1: Decision Stump")
        sys.exit(1)
    
    data_path = sys.argv[1]
    target_path = sys.argv[2]
    base_type = int(sys.argv[3])
    if base_type not in [0, 1]:
        print("错误: base_type 必须是 0 (Logistic Regression) 或 1 (Decision Stump)")
        sys.exit(1)

    for path in [data_path, target_path]:
        if not os.path.exists(path):
            print(f"错误: 文件不存在 - {path}")
            sys.exit(1)
#交叉验证
    try:
        X, y = load_data(data_path, target_path)
        output_dir = os.path.join('experiments')
        run_cross_validation(X, y, base_type, output_dir)
    except Exception as e:
        print(f"执行过程中发生错误: {str(e)}")
        sys.exit(1)

if __name__ == '__main__':
    main()