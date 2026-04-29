import numpy as np

class AdaBoost:
    def __init__(self, base_classifier, n):
        self.base_classifier = base_classifier  # 基学习器，有LogisticRegression和DecisionStamp两种
        self.n = n  # 基分类器数量
        self.classifiers = []  # 存储训练好的基分类器
        self.classifier_weights = []  # 基学习器权重

    def fit(self, X, y):
        m = X.shape[0]
        weights = np.ones(m) / m  # 初始化样本权重为均匀分布，将原始标签转换为0/1格式
        y = y.astype(int)
        
        # 添加提前终止条件，处理完美分类情况
        perfect_classification = False
        for _ in range(self.n):
            #归一化样本权重，确保权重之和为 1
            weights /= np.sum(weights)
            #训练基分类器
            classifier = self.base_classifier()
            classifier.fit(X, y, weights)
            predictions = classifier.predict(X).astype(int)  # 确保输出0/1标签

            #计算错误率
            error = np.sum(weights * (predictions != y))
            #处理错误率大于0.5的情况
            inverted = False
            if error > 0.5:
                predictions = 1 - predictions  #取反
                error = 1 - error
                inverted = True
            
            #注意特殊情况哦
            if error < 1e-10:
                alpha = 1e10  # 设置极大权重，不能除以0呀
                perfect_classification = True
            else:
                alpha = 0.5 * np.log((1 - error) / error)  #计算分类器权重
                
            # 更新样本权重：错误分类的样本权重增加，正确分类的减少
            # 使用高效的点积计算代替逐元素乘法
            weights_update = np.exp(-alpha * (2 * (y == predictions) - 1))
            weights *= weights_update
                
            # 存储训练好的分类器及其权重
            self.classifiers.append((classifier, inverted))
            self.classifier_weights.append(alpha)
            
            # 完美分类时提前终止训练
            if perfect_classification:
                break

    def predict(self, X):
        if not self.classifiers:
            return np.zeros(X.shape[0], dtype=int)
        # 计算所有基学习器的加权和
        aggregate = np.zeros(X.shape[0])
        total_alpha = np.sum(self.classifier_weights)  # 计算权重总和一次
        for (classifier, inverted), alpha in zip(self.classifiers, self.classifier_weights):
            normalized_alpha = alpha / total_alpha
            pred = classifier.predict(X).astype(int)
            if inverted:
                pred = 1 - pred       
            aggregate += normalized_alpha * pred
        # 将加权和转换为0/1标签，阈值为0.5
        return (aggregate >= 0.5).astype(int)