import os
import numpy as np
import pandas as pd
import torch
from torch import nn

# 对cifar10定义卷积神经网络
class CNN(nn.Module):
    def __init__(self, num_classes=10, kernel_size=5):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, kernel_size)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, kernel_size)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, num_classes)

        self.sequntial = nn.Sequential(
            self.conv1,
            nn.ReLU(),
            self.pool,
            self.conv2,
            nn.ReLU(),
            self.pool,
            nn.Flatten(),
            self.fc1,
            nn.ReLU(),
            self.fc2,
            nn.ReLU(),
            self.fc3
        )