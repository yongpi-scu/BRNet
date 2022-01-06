import torch
import torch.nn.functional as F
from torch.nn.modules.loss import _Loss
from torch import nn

class OrderedLoss(_Loss):
    def __init__(self, alpha=1, beta=0.5):
        super(OrderedLoss, self).__init__()
        self.criterion = nn.CrossEntropyLoss()
        self.alpha = alpha
        self.beta = beta
        self.epsilon = 0.01 # avoid log(0)

    def forward(self, x, target):
        loss_1 = self.criterion(x, target)
        x = F.softmax(x, dim=1)
        loss_2 = -(x.argmax(1).float()-target.float())**2*(torch.log(1-x.max(1)[0]+self.epsilon))
        loss = self.alpha*loss_1+self.beta*loss_2.mean()
        return loss