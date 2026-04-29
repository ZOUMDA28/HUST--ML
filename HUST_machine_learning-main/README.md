# HUST_machine_learning:AdaBoost 机器学习项目

基于 AdaBoost 集成学习算法的二分类实现，支持逻辑回归和决策树桩两种基分类器。

## 项目简介

本项目是机器学习课程的结课作业，实现了 AdaBoost 算法用于二分类任务。通过集成多个弱分类器，逐步提升模型的整体预测性能。项目采用 10 折交叉验证评估模型表现。

## 项目结构

```
├── data/                          # 数据目录
│   ├── data.csv                   # 特征数据
│   └── targets.csv                # 标签数据
│
├── experiments/                   # 实验结果目录
│   └── base#_fold#.csv            # 各配置下的预测结果
│
├── adaboost.py                    # AdaBoost 算法核心实现
├── base_classifiers.py            # 基分类器实现（逻辑回归、决策树桩）
├── utils.py                       # 工具函数（数据加载、K折划分）
├── main.py                        # 程序入口
├── evaluate.py                    # 模型评估脚本
│
├── 预测结果.xlsx                   # 预测结果汇总
└── README.md                      # 项目说明文档
```

## 算法说明

### AdaBoost 算法

AdaBoost (Adaptive Boosting) 是一种自适应增强算法，核心思想是：

1. 初始化样本权重为均匀分布
2. 训练基分类器并计算错误率
3. 根据错误率计算分类器权重 α
4. 更新样本权重：错误分类样本权重增加，正确分类样本权重减少
5. 重复上述过程，最终通过加权投票得到最终预测

### 基分类器

#### 1. 逻辑回归 (Logistic Regression)
- 使用梯度下降优化
- 支持样本权重
- 包含数据标准化处理

#### 2. 决策树桩 (Decision Stump)
- 单层决策树
- 遍历所有特征和阈值寻找最优分割点
- 计算加权错误率选择最佳划分

## 环境要求

- Python 3.x
- NumPy
- Pandas
- Scikit-learn

安装依赖：
```bash
pip install numpy pandas scikit-learn
```

## 使用方法

### 运行主程序

```bash
python main.py ./data/data.csv ./data/targets.csv [base_type]
```

参数说明：
- `base_type`: 基分类器类型
  - `0`: 逻辑回归
  - `1`: 决策树桩

示例：
```bash
# 使用逻辑回归作为基分类器
python main.py ./data/data.csv ./data/targets.csv 0

# 使用决策树桩作为基分类器
python main.py ./data/data.csv ./data/targets.csv 1
```

### 评估结果

运行评估脚本查看各配置下的平均准确率：
```bash
python evaluate.py
```

## 实验配置

项目对每种基分类器分别测试了以下配置：

| 基分类器数量 | 说明 |
|-------------|------|
| 1 | 单一基分类器 |
| 5 | 5个基分类器集成 |
| 10 | 10个基分类器集成 |
| 100 | 100个基分类器集成 |

每种配置均采用 10 折交叉验证评估。

## 输出说明

程序运行后，预测结果保存在 `experiments/` 目录下：

- 文件命名格式：`base{n}_fold{k}.csv`
- 内容格式：`Index,Prediction`（索引从1开始）

## 作者

- 学号：I202420199
- 姓名：MARTIN

