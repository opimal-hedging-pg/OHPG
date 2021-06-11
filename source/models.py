# -*- coding: utf-8 -*-

import os

import torch
import torch.nn as nn

from .utils.mlp import MlpConstructor

class PolicyModel(nn.Module):
    """
    policy network
    """
    @property
    def name(self):
        return 'PolicyL5Tanh'

    def __init__(self, dim_states = 5, n_actions = 20, init = True):
        
        super().__init__()

        self.n_actions = n_actions

        self._model = MlpConstructor(dim_states, n_actions, [100, 100, 100, 100], 'tanh')

        if init:
            for m in self._model:
                if isinstance(m, nn.Linear):
                    m.weight.data.normal_(0, 1.)
                else:
                    pass
        else:
            pass

        self.out = nn.Softmax(-1)

        self._model = nn.Sequential(self._model, self.out)

    def forward(self, *args, **kwargs):
        return self._model.forward(*args, **kwargs)

class RealPolicyModel(nn.Module):
    """
    policy network only for sp500 index
    """
    @property
    def name(self):
        return 'RealPolicyL1Tanh'

    def __init__(self, dim_states = 5, n_actions = 3, init = True):
        
        super().__init__()

        self.n_actions = n_actions

        self._model = MlpConstructor(dim_states, n_actions, [100], 'tanh')

        if init:
            for m in self._model:
                if isinstance(m, nn.Linear):
                    m.weight.data.normal_(0, 1.)
                else:
                    pass
        else:
            pass

        self.out = nn.Softmax(-1)

        self._model = nn.Sequential(self._model, self.out)

    def forward(self, *args, **kwargs):
        return self._model.forward(*args, **kwargs)


class ValueModel(nn.Module):
    """
    value function network
    """
    @property
    def name(self):
        return 'ValueL1Tanh'

    def __init__(self, dim_states = 5, init = True):
        super().__init__()
        self._model = MlpConstructor(dim_states, 1, [100, 100, 100], 'tanh')

        if init:
            for m in self._model:
                if isinstance(m, nn.Linear):
                    m.weight.data.normal_(0, 1.)
                else:
                    pass
        else:
            pass

    def forward(self, *args, **kwargs):
        return self._model.forward(*args, **kwargs)

class NewPolicyModel(PolicyModel):
    """
    policy network adapted to NewEnv and NewPipe.
    """
    @property
    def name(self):
        return 'NewPolicyL3Tanh'

    def __init__(self, dim_states = 5, n_actions = 3, init = True):
        
        super().__init__(dim_states, n_actions, init)







