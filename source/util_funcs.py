# -*- coding: utf-8 -*-


import numpy as np
import torch

class BaseUtility(object):
    
    @property
    def name(self):
        raise NotImplementedError()


class linear_utility(BaseUtility):
    
    @property
    def name(self):
        return 'Ulin'
    
    def __init__(self, *args, **kwargs):
        pass
    
    def __call__(self, x):
        return x

class lingau_utility(BaseUtility):
    
    @property
    def name(self):
        rp = '_'.join(str(self.rp).split('.'))
        return f'Lingau-{rp}'
    
    def __init__(self, risk_pref = 1.):
        self.rp = risk_pref
    
    def __call__(self, x):
        if abs(x) <= self.rp:
            return 10.
        else:
            return -abs(x) + self.rp

    
class exp_utility(BaseUtility):
    
    @property
    def name(self):
        rp = '_'.join(str(self.rp).split('.'))
        return f'Uexp-{rp}'
    
    def __init__(self, risk_pref = 1.):
        self.rp = risk_pref
    
    def __call__(self, x):
        
        if self.rp == 0:
            return x
        else:
            return (1 - torch.exp(- torch.clip(self.rp * x, -100, 100))) / self.rp
        

class gaussian_utility(BaseUtility):
    
    """
    TODO: Provide normalization?
    """
    
    @property
    def name(self):
        rp = '_'.join(str(self.rp).split('.'))
        return f'Ugau-{rp}'
    
    def __init__(self, risk_pref = .5):
        assert risk_pref > 0
        self.rp = risk_pref 
        
    def __call__(self, x):
        return torch.exp(- (x ** 2) / (2 * self.rp ** 2))
    
    
class epsilon_utility(BaseUtility):
    
    """
    Counter, if |x| is smaller than a threshold.
    
    TODO: Provide normalization?
    """
    
    @property
    def name(self):
        rp = '_'.join(str(self.rp).split('.'))
        return f'Ueps-{rp}'
    
    def  __init__(self, epsilon = .1, normalize = True):
        assert epsilon >= 0.
        self.rp = epsilon
        
        
    def __call__(self, x):
        return torch.sum(torch.abs(x) <= self.rp)


class crra_utility(BaseUtility):
    
    """
    isoelastic utility.
    only for risk-aversion and risk-neutral?
    """
    
    @property
    def name(self):
        rp = '_'.join(str(self.rp).split('.'))
        return f'Ucrra-{rp}'
    
    def __init__(self, risk_pref = 1.):
        assert risk_pref >= 0. 
        self.rp = risk_pref

    def __call__(self, x):
        
        if self.rp == 1.:
            return torch.log(x)
        else:
            return (x ** (1 - self.rp) - 1) / (1 - self.rp)


class crra_extended(crra_utility):
    
    @property
    def name(self):
        rp = '_'.join(str(self.rp).split('.'))
        return f'Ucrraext-{rp}'

    def __init__(self, risk_pref = 1.):
        super().__init__(risk_pref = risk_pref)

    def __call__(self, x):

        if x < 0:
            return x
        else:
            return super().__call__(x)

class absol_utility(BaseUtility):
    
    @property
    def name(self):
        return 'Uabs'

    def __init__(self, *args, **kwargs): 
        pass
    
    def __call__(self, x):
        return - torch.abs(x)

class mean_var_utility(BaseUtility):

    @property
    def name(self):
        rp = '_'.join(str(self.rp).split('.'))
        return f'MeanVar-{rp}'

    def __init__(self, risk_pref = 1.):
        # super().__init__(risk_pref = risk_pref)
        self.rp = risk_pref

    def __call__(self, x):
        return (x - self.rp * x ** 2)


class quad_utility(BaseUtility):

    @property
    def name(self):
        rp = '_'.join(str(self.rp).split('.'))
        return f'Quad-{rp}'

    def __init__(self, risk_pref = 1.):
        self.rp = risk_pref


    def __call__(self, x):
        return - (x ** 2) / self.rp
        
        


