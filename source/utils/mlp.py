# -*- coding: utf-8 -*-


import os

import torch
from torch import nn



def MlpConstructor(in_dim, final_dim, hidden_config, nonlinear = 'relu'):
    
    
    in_dim = int(in_dim)
    
    modules = []
    
    # Hidden layers
    
    for out_dim in hidden_config:
        
        modules.append(nn.Linear(in_dim, out_dim))

        if nonlinear == 'relu':
            modules.append(nn.ReLU())
        elif nonlinear == 'leakyrelu':
            modules.append(nn.LeakyReLU())
        elif nonlinear == 'sigmoid':
            modules.append(nn.Sigmoid())
        elif nonlinear == 'tanh':
            modules.append(nn.Tanh())
        else:
            raise NotImplementedError(f'Not implemented nonlinear: {nonlinear}')
        
        modules.append(nn.LayerNorm(out_dim))
        
        in_dim = out_dim
        
    # Final out layer
    
    modules.append(nn.Linear(in_dim, final_dim, bias = True))
    
    return nn.Sequential(*modules)