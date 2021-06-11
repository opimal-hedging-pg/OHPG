
__all__ = ['BS_delta', 'BS_reward', 'BS_final_wealth_']

import numpy as np
import torch
from scipy.stats import norm

from .. import util_funcs
from .. import trancost_funcs

def BS_delta(S, K, T, r):
    
    """
    Black Scholes delta
    """
    
    logS = np.log(S)
    y = logS[: , 1:] - logS[: , 0 : logS.shape[1] - 1]  
    dt = 1
    sigma = np.std(y, axis = 1) / np.sqrt(dt)    
    
    delta = np.zeros(y.shape)
    
    for i in range(y.shape[0]):
        for j in range(y.shape[1]):
    
            d1 = (logS[i, j] - np.log(K) + (r + sigma[i] ** 2 / 2) * (T - j)) / (sigma[i] * np.sqrt(T - j))
    
            delta[i, j] = norm.cdf(d1)
    
    return delta                


def BS_reward(S, Z, B0, K, T, r, dt = 1., grids = None, utility = None, transaction = None, qlbs_reward = True, delta_out = True):
    '''
    for single path.
    under 3d state reward, transition cost is useless.
    '''
    if utility is not callable:
        utility = util_funcs.exp_utility(risk_pref = 0.)
    else:
        pass
    if transaction is not callable:
        transaction = trancost_funcs.constant_trancost(tc_para = 0.)
    else:
        pass

    delta = np.squeeze(BS_delta(S[np.newaxis, :], K, T, r))
    a_bs = delta - np.append(0, delta[:-1])
    if grids is None:
        pass
    else:
        if delta_out:
            # discretize delta
            for i in range(len(delta)):
                dis = np.abs(delta[i] - grids)
                delta[i] = grids[np.argmin(dis)]
                a_bs = delta - np.append(0, delta[:-1])
        else:
            # discretize action
            for i in range(len(a_bs)):
                dis = np.abs(a_bs[i] - grids)
                a_bs[i] = grids[np.argmin(dis)]
                delta = np.cumsum(a_bs)

    B = [B0]
    for a,s in zip(a_bs, S[:-1]):
        B_ep = (B[-1] - a * s - transaction(a * s)) * np.exp(r * dt)
        B.append(B_ep)

    B = np.array(B)
    pv_bs = B + np.append(0., delta) * S
    
    if qlbs_reward:
        re = delta * (S[1:] * np.exp(-r * dt) - S[:-1])
    else:
        re = (B[1:] - B[:-1]) + (delta * S[1:] - np.append(0, delta[:-1]) * S[:-1]) - (Z[1:] - Z[:-1])
    gamma = np.exp(-r * dt)
    discount = gamma ** np.arange(0, len(re), 1)
    re *= discount
    reward_bs = torch.sum(utility(torch.Tensor(re)))
    cum_rd_bs = torch.cumsum(utility(torch.Tensor(re)), dim = 0)
    cum_rd_bs = torch.cat((torch.Tensor([0.]), cum_rd_bs))
    
    return reward_bs.numpy(), a_bs, delta, pv_bs, cum_rd_bs.numpy()

def BS_final_wealth_(S, Z, B0, K, T, r, trancost, grids = None, delta_out = True):
    
    if trancost is not callable:
        trancost = trancost_funcs.constant_trancost(tc_para = 0.)
    else:
        pass
    
    delta = np.squeeze(BS_delta(S[np.newaxis, :], K, T, r))
    a = delta - np.append(0, delta[0 : len(delta)-1])
    if grids is None:
        pass
    else:
        if delta_out:
            # discretize delta
            for i in range(len(delta)):
                dis = np.abs(delta[i] - grids)
                delta[i] = grids[np.argmin(dis)]
                a = delta - np.append(0, delta[0 : len(delta)-1])
        else:
            # discretize action
            for i in range(len(a)):
                dis = np.abs(a[i] - grids)
                a[i] = grids[np.argmin(dis)]
                delta = np.cumsum(a)

    B = B0 * np.exp(r * T)
    S_ = delta[-1] * S[-1]
    O = np.sum(a * S[0:T] * np.exp(r * np.arange(T, 0, -1)))

    tc = np.sum(trancost(a * S[0:T]) * np.exp(r * np.arange(T, 0, -1)))
    
    return B + S_ - O - tc - Z[-1]