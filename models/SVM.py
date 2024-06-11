import numpy as np
import torchvision.transforms as transforms
import pickle



# 提取数据并展平图像
def extract_data(dataloader):
    for data in dataloader:
        images, labels = data
    return images.numpy().reshape(images.shape[0], -1), labels.numpy()


# SVM 训练和预测
class SVM:
    def __init__(self, kernel='linear', C=1.0, gamma=None, degree=3):
        self.kernel = kernel
        self.C = C
        self.gamma = gamma
        self.degree = degree
        self.alpha = None
        self.b = None
        self.X = None
        self.y = None

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.alpha = np.zeros(n_samples)
        self.b = 0.0
        self.X = X
        self.y = y
        K = self.compute_kernel(X)

        # Simplified SMO algorithm
        for sss in range(100):
            for i in range(n_samples):
                error_i = self.predict_single(X[i]) - y[i]
                if (y[i] * error_i < -0.001 and self.alpha[i] < self.C) or (y[i] * error_i > 0.001 and self.alpha[i] > 0):
                    j = np.random.randint(0, n_samples)
                    error_j = self.predict_single(X[j]) - y[j]
                    alpha_i_old, alpha_j_old = self.alpha[i], self.alpha[j]
                    if y[i] != y[j]:
                        L = max(0, self.alpha[j] - self.alpha[i])
                        H = min(self.C, self.C + self.alpha[j] - self.alpha[i])
                    else:
                        L = max(0, self.alpha[i] + self.alpha[j] - self.C)
                        H = min(self.C, self.alpha[i] + self.alpha[j])
                    if L == H:
                        continue
                    eta = 2 * K[i, j] - K[i, i] - K[j, j]
                    if eta >= 0:
                        continue
                    self.alpha[j] -= y[j] * (error_i - error_j) / eta
                    self.alpha[j] = np.clip(self.alpha[j], L, H)
                    if abs(self.alpha[j] - alpha_j_old) < 1e-5:
                        continue
                    self.alpha[i] += y[i] * y[j] * (alpha_j_old - self.alpha[j])
                    b1 = self.b - error_i - y[i] * (self.alpha[i] - alpha_i_old) * K[i, i] - y[j] * (self.alpha[j] - alpha_j_old) * K[i, j]
                    b2 = self.b - error_j - y[i] * (self.alpha[i] - alpha_i_old) * K[i, j] - y[j] * (self.alpha[j] - alpha_j_old) * K[j, j]
                    if 0 < self.alpha[i] < self.C:
                        self.b = b1
                    elif 0 < self.alpha[j] < self.C:
                        self.b = b2
                    else:
                        self.b = (b1 + b2) / 2

    def compute_kernel(self, X):
        if self.kernel == 'linear':
            return np.dot(X, X.T)
        elif self.kernel == 'poly':
            return (np.dot(X, X.T) + 1) ** self.degree
        elif self.kernel == 'rbf':
            if self.gamma is None:
                self.gamma = 1 / X.shape[1]
            K = np.zeros((X.shape[0], X.shape[0]))
            for i in range(X.shape[0]):
                K[i] = np.exp(-self.gamma * np.sum((X[i] - X) ** 2, axis=1))
            return K
        else:
            raise ValueError("Unsupported kernel")

    def predict_single(self, x):
        if self.kernel == 'linear':
            return np.dot(x, (self.alpha * self.y) @ self.X) + self.b
        elif self.kernel == 'poly':
            return (np.dot(x, (self.alpha * self.y) @ self.X) + 1) ** self.degree + self.b
        elif self.kernel == 'rbf':
            if self.gamma is None:
                self.gamma = 1 / self.X.shape[1]
            K = np.exp(-self.gamma * np.sum((x - self.X) ** 2, axis=1))
            return np.dot(K, self.alpha * self.y) + self.b
        else:
            raise ValueError("Unsupported kernel")

    def predict(self, X):
        return np.sign(np.array([self.predict_single(x) for x in X]))



