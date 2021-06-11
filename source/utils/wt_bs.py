__all__ = ['WT_BS']
import numpy as np
from . import BS_final_wealth_


def WT_BS(data, meta, trancost = None, grids = None, delta_out = True):
    '''
    final wealth under continuous BS delta
    data: N * T+1 * 2
    '''
    data = np.array(data)
    wtbs = []

    if isinstance(meta, dict):
        for dat in data:
            wtbs.append(BS_final_wealth_(S = dat[:, 0],
                                         Z = dat[:, 1],
                                         B0 = dat[0, 1],
                                         K = meta['K'],
                                         T = meta['T'],
                                         r = meta['r'],
                                         trancost = trancost,
                                         grids = grids,
                                         delta_out = delta_out))
    
    else:
        assert isinstance(meta, np.ndarray), 'meta is not numpy array'
        assert len(meta) == data.shape[0], 'meta and data are not one-to-one'
        for met, dat in zip(meta, data):

            wtbs.append(BS_final_wealth_(S = dat[:, 0],
                                         Z = dat[:, 1],
                                         B0 = dat[0, 1],
                                         K = met['K'],
                                         T = met['T'],
                                         r = met['r'],
                                         trancost = trancost,
                                         grids = grids,
                                         delta_out = delta_out))
    
        

    return np.array(wtbs)