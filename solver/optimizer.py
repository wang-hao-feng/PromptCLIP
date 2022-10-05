import torch
import torch.nn as nn
from torch.optim import SGD

def make_optimizer(model:nn.Module, lr:float, weight_decky:float, momentum:float=0.9):
    params = []
    for key, value in model.named_parameters():
        if value.requires_grad:
            params += [{
                'params': [value],
                'lr': lr, 
                'weight_decay': weight_decky
            }]
    optimizer = SGD(params=params, lr=lr, momentum=momentum, weight_decay=weight_decky)
    return optimizer