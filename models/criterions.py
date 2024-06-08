import torch
import torch.nn as nn

class LabelSmoothingCrossEntropyLoss(nn.Module):
    def __init__(self, classes, smoothing=0.0, dim=-1):
        super(LabelSmoothingCrossEntropyLoss, self).__init__()
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.cls = classes
        self.dim = dim

    def forward(self, pred, target):
        pred = pred.log_softmax(dim=self.dim)
        with torch.no_grad():
            true_dist = torch.zeros_like(pred)
            true_dist.fill_(self.smoothing / (self.cls - 1))
            true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
        return torch.mean(torch.sum(-true_dist * pred, dim=self.dim))    

class HingeL2Loss(nn.Module):
    def __init__(self, classes):
        super(HingeL2Loss, self).__init__()
        self.cls = classes
    
    def forward(self, pred, target):
        hinge_loss = torch.mean(torch.max(1 - pred.t() * (2*target.float()-1), torch.tensor(0.0).to(pred.device)))
        l2_loss = torch.mean(torch.norm(pred, p=2, dim=1))
        return hinge_loss + 0.01*l2_loss