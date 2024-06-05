import os
import numpy as np
import pandas as pd
import torch
from torch import nn

# 定义不同的核函数：线性、gaussian、多项式
def linear_kernel(x):
    return x

def gaussian_kernel(x, gamma=1):
    n = x.shape[0]
    x = x.unsqueeze(0).expand(n, -1, -1)
    y = x.transpose(0, 1)
    return torch.exp(-gamma * torch.sum((x - y) ** 2, dim=2))

def polynomial_kernel(x, p=2):
    return (torch.mm(x, x.t()) + 1) ** p


# 构建SVM模型
class SVM(nn.Module):
    def __init__(self, input_dim, kernel=None):
        super(SVM, self).__init__()
        self.kernel = kernel
        self.w = nn.Parameter(torch.randn(input_dim, 1))
        self.b = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        if self.kernel is not None:
            x = self.kernel(x)
        return torch.mm(x, self.w) + self.b

    def loss(self, x, y):
        y_hat = self.forward(x)
        hinge_loss = torch.mean(torch.clamp(1 - y * y_hat, min=0))
        l2_loss = torch.sum(self.w ** 2) / 2
        return hinge_loss + 1e-3 * l2_loss
    
    def predict(self, x):
        return torch.argmax(self.forward(x), dim=1)
    
    def predict_proba(self, x):
        return self.forward(x).squeeze().detach().numpy()

