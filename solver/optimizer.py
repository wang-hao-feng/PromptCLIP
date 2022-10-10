import sys
sys.path.append('..')

import torch.nn as nn
from torch.optim import SGD, Adam
from prompt_clip import PromptCLIP

def make_optimizer(model:PromptCLIP, lr:float, weight_decky:float, optim:str='SGD', momentum:float=0.9, betas:tuple=(0.9, 0.999), amsgrad:bool=False):
    params = []
    params_size = 0
    def calsize(size):
        s = 1
        for l in size:
            s *= l
        return s
    for key, value in model.named_parameters():
        if value.requires_grad:
            params += [{
                'params': [value],
                'lr': lr, 
                'weight_decay': weight_decky
            }]
            params_size += calsize(value.size())
    print(f'Learnable parameters size: {params_size / 1e3}K')
    optimizer = None
    if optim == 'SGD':
        optimizer = SGD(params=params, lr=lr, momentum=momentum, weight_decay=weight_decky)
    elif optim == 'Adam':
        optimizer = Adam(params=params, lr=lr, betas=betas, weight_decay=weight_decky, amsgrad=amsgrad, eps=1e-3)
    return optimizer