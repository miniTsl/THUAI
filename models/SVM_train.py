import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
import pickle
from SVM import SVM, extract_data

# 数据加载和预处理
transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=False, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=10000, shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=False, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=10000, shuffle=False, num_workers=2)

X_train, y_train = extract_data(trainloader)
X_test, y_test = extract_data(testloader)

# 使用不同核训练SVM
svm = SVM(kernel='rbf', C=1.0)

svm.fit(X_train, y_train)

# 预测并计算准确率
predictions = svm.predict(X_test)
accuracy = np.mean(predictions == y_test)
print(f"Test Accuracy: {accuracy * 100:.2f}%")

# 存储模型参数
with open('svm_10_rbf.pkl', 'wb') as f:
    pickle.dump(svm, f)