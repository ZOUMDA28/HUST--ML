import numpy as np
from utils import sigmoid #导入sigmoid



class LogisticRegression:
    def __init__(self, learning_rate=0.1, max_iter=100, tol=1e-4):
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.tol = tol  # 添加收敛阈值
        self.weights = None
        self.bias = None
        self.mean = None
        self.std = None

    def standardize(self, X):
        if self.mean is None or self.std is None:
            self.mean = np.nanmean(X, axis=0)  #处理可能存在的NaN值
            self.std = np.nanstd(X, axis=0)
            self.std[self.std == 0] = 1  #防止除零
        return (X - self.mean) / self.std
    
    def fit(self, X, y, sample_weights=None):
        n_samples, n_features = X.shape
        
        # 初始化参数
        self.weights = np.zeros(n_features)
        self.bias = 0
        
        # 处理样本权重
        if sample_weights is None:
            sample_weights = np.ones(n_samples)
        sample_weights = sample_weights / np.sum(sample_weights)  #归一化权重
        

        
        # 梯度下降
        for i in range(self.max_iter):
            linear_model = np.dot(X, self.weights) + self.bias
            y_pred = self._sigmoid(linear_model)
            
            error = y_pred - y
            dw = np.dot(X.T, error * sample_weights)  # 移除1/n_samples因子
            db = np.sum(error * sample_weights)
            
            gradient_norm = np.linalg.norm(np.concatenate((dw, [db])))
            if gradient_norm < self.tol:
                break
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db

    def predict(self, X):
        X = self.standardize(X)
        linear_model = np.dot(X, self.weights) + self.bias
        return (self._sigmoid(linear_model) >= 0.5).astype(int)

    def _sigmoid(self, z):
        z = np.clip(z, -500, 500)  #防止数值溢出
        return 1 / (1 + np.exp(-z))

class DecisionStump:
    def __init__(self):
        self.feature_idx = None  #特征值
        self.threshold = None    
        self.left_label = 1      #特征值<=阈值时的要打的标签
        self.right_label = 0     #特征值>阈值时的标签

    def fit(self, X, y, sample_weights):
        n_samples, n_features = X.shape
        best_error = float('inf')
        
        # 遍历所有特征
        for feature_idx in range(n_features):
            feature_values = np.sort(np.unique(X[:, feature_idx]))
            
            #尝试每个唯一值作为候选阈值
            for threshold in feature_values:
                # 计算布尔掩码
                mask = X[:, feature_idx] <= threshold
                # 尝试两种标签组合1/0和0/1
                for left_label, right_label in [(0, 1), (1, 0)]:
                    pred = np.where(mask, left_label, right_label)
                    #计算加权错误率
                    error = np.sum(sample_weights * (pred != y))
                    #更新错误率信息
                    if error < best_error:
                        best_error = error
                        self.feature_idx = feature_idx
                        self.threshold = threshold
                        self.left_label = left_label
                        self.right_label = right_label
                        #如果达到完美分类则退出
                        if np.isclose(error, 0):
                            return

    def predict(self, X):
        mask = X[:, self.feature_idx] <= self.threshold
        return np.where(mask, self.left_label, self.right_label)