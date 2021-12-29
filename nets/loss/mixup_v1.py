import os
import sys
import time
import math
import numpy as np
import torch

def mixup_data(x, y, alpha=1.0):

    '''Compute the mixup data. Return mixed inputs, pairs of targets, and lambda'''
    lam = np.random.beta(alpha, alpha)
    batch_size = x.size()[0]
    index = torch.randperm(batch_size).to(x.device)
    mixed_x = lam * x + (1 - lam) * x[index,:]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam

def mixup_criterion(y_a, y_b, lam):
    return lambda criterion, pred: (lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)).mean()