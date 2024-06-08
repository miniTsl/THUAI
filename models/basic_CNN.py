import torch
import torch.nn as nn

# 对cifar10定义卷积神经网络
class basic_CNN(nn.Module):
    def __init__(self, in_c, num_classes=10, kernel_size=3):
        super(basic_CNN, self).__init__()
        self.conv1 = nn.Conv2d(in_c, 6, kernel_size)
        self.batchnorm_1 = nn.BatchNorm2d(6)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, kernel_size)
        self.batchnorm_2 = nn.BatchNorm2d(16)
        self.fc1 = nn.Linear(16 * 6 * 6, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, num_classes)

        self.sequntial = nn.Sequential(
            self.conv1,
            self.batchnorm_1,
            self.pool,
            nn.ReLU(),
            nn.Dropout(0.1),
            self.conv2,
            self.batchnorm_2,
            self.pool,
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Flatten(),
            self.fc1,
            nn.ReLU(),
            self.fc2,
            nn.ReLU(),
            self.fc3
        )

    def forward(self, x):
        return self.sequntial(x)
    
    def predict(self, x):
        return torch.argmax(self.forward(x), dim=1)
    
    def predict_proba(self, x):
        return self.forward(x).squeeze().detach().numpy()
 

# model = basic_CNN(3, 10)
# print(model)