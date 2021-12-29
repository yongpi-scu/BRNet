import os
import sys
import time
import math
import numpy as np
import torch

def mixup_data(x, y, alpha=1.0):

    '''Compute the mixup data. Return mixed inputs, pairs of targets, and lambda'''
    batch_size = x.size()[0]
    lambd = np.random.beta(alpha, alpha, batch_size)
    lambd = np.concatenate([lambd[:,None], 1-lambd[:,None]], 1).max(1)
    lambd = x.new(lambd)
    shuffle = torch.randperm(x.size(0)).to(x.device)
    x1, y1 = x[shuffle], y[shuffle]
    out_shape = [lambd.size(0)] + [1 for _ in range(len(x.shape) - 1)]
    mixed_x = (x * lambd.view(out_shape) + x1 * (1-lambd).view(out_shape))
    y_a, y_b = y, y[shuffle]
    return mixed_x, y_a, y_b, lambd

def mixup_criterion(y_a, y_b, lam):
    return lambda criterion, pred: (lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)).mean()
