__all__ = ['re_bs']
import torch
import numpy as np
from . import BS_reward

def re_bs(data, meta, grids = None, utility = None, trancost = None, qlbs_reward = True, delta_out = True):
    '''
    cumulative reward of black scholes delta.
    data: N * T+1 * 2
    '''
    data = np.array(data)
    re_bs = []
    a_bs = []
    delta_bs = []
    pv_bs = []
    cum_rd_bs = []

    if isinstance(meta, dict):
        for dat in data:
            re_bs_ep, a_bs_ep, delta_bs_ep, pv_bs_ep, cum_rd_bs_ep = BS_reward(S = dat[:, 0], 
                                                                               Z = dat[:, 1], 
                                                                               B0 = dat[0, 1], 
                                                                               K = meta['K'], 
                                                                               T = meta['T'], 
                                                                               r = meta['r'], 
                                                                               dt = meta['dt'], 
                                                                               grids = grids, 
                                                                               utility = utility, 
                                                                               transaction = trancost,
                                                                               qlbs_reward = qlbs_reward,
                                                                               delta_out = delta_out)
            re_bs.append(re_bs_ep)
            a_bs.append(a_bs_ep)
            delta_bs.append(delta_bs_ep)
            pv_bs.append(pv_bs_ep)
            cum_rd_bs.append(cum_rd_bs_ep)
    else:
        assert isinstance(meta, np.ndarray), 'meta is not numpy array'
        assert len(meta) == data.shape[0], 'meta and data are not one-to-one'
        for met, dat in zip(meta, data):
            re_bs_ep, a_bs_ep, delta_bs_ep, pv_bs_ep, cum_rd_bs_ep = BS_reward(S = dat[:, 0], 
                                                                               Z = dat[:, 1], 
                                                                               B0 = dat[0, 1], 
                                                                               K = met['K'], 
                                                                               T = met['T'], 
                                                                               r = met['r'], 
                                                                               dt = met['dt'], 
                                                                               grids = grids, 
                                                                               utility = utility, 
                                                                               transaction = trancost,
                                                                               qlbs_reward = qlbs_reward,
                                                                               delta_out = delta_out)
            re_bs.append(re_bs_ep)
            a_bs.append(a_bs_ep)
            delta_bs.append(delta_bs_ep)
            pv_bs.append(pv_bs_ep)
            cum_rd_bs.append(cum_rd_bs_ep)
    
            
    return np.squeeze(re_bs), np.squeeze(a_bs), np.squeeze(delta_bs), np.squeeze(pv_bs), np.squeeze(cum_rd_bs)




