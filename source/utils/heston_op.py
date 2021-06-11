
__all__ = ['Heston_price']

import numpy as np 
import math
from scipy.integrate import quad
import time


def Heston_price(kappa, theta, sigma, rho, S, V, K, r, tau, dt = None):
    '''
    option price under Heston model.
    S,V,tau must have the same length.
    r is per dt and T = tau * dt.
    '''
    ht_prices = []
    assert S.shape == tau.shape, 'shape is different'
    for s, v, tau_ in zip(S, V, tau):
        P = np.zeros(2)
        integrand_p1 = lambda phy: np.real(np.exp(complex(0,-1) * phy * np.log(K)) * myf_j(phy, 0, kappa, theta, sigma, rho, s, v, r, tau_) / (phy * complex(0,1)))
        integrand_p2 = lambda phy: np.real(np.exp(complex(0,-1) * phy * np.log(K)) * myf_j(phy, 1, kappa, theta, sigma, rho, s, v, r, tau_) / (phy * complex(0,1)))

        P[0] = 0.5 + (1 / np.pi) * quad(integrand_p1, 0, np.inf)[0]
        P[1] = 0.5+(1 / np.pi) * quad(integrand_p2, 0, np.inf)[0]

        C = s * P[0] - K * np.exp(-r * tau_) * P[1]
        assert math.isnan(C) is not True, 'option price is NAN'
        ht_prices.append(C)
        

    return np.squeeze(np.array(ht_prices))


def myf_j(phy, j, kappa, theta, sigma, rho, S, V, r, tau):
    '''
    f_j(phy) in heston solution.
    j = 0,1  w.r.t P1 and P2.
    '''
   
    mu = [complex(0.5), complex(-0.5)]
    a = complex(kappa * theta)
    b = [complex(kappa - rho * sigma), complex(kappa)]
    
    d = [complex(0,0),complex(0,0)]
    g = [complex(0,0),complex(0,0)]
    C = [complex(0,0),complex(0,0)]
    D = [complex(0,0),complex(0,0)]
    f = [complex(0,0),complex(0,0)]
    
    for k in range(2):
        d[k] = np.sqrt(np.square(b[k] - rho * sigma * phy * complex(0,1)) - np.square(sigma) * (2 * phy * mu[k] * complex(0,1) - np.square(phy)))
        g[k] = (b[k] - rho * sigma * phy * complex(0,1) + d[k]) / (b[k] - rho * sigma * phy * complex(0,1) - d[k])
        C[k] = r * complex(0,1) * phy * tau + a * ((b[k] - rho * sigma * phy * complex(0,1) + d[k]) * tau - 2 * np.log((1 - g[k] * np.exp(d[k] * tau)) / (1 - g[k]))) / np.square(sigma)
        D[k] = ((b[k] - rho * sigma * phy * complex(0,1) + d[k]) / sigma ** 2) * ((1 - np.exp(d[k] * tau)) / (1 - g[k] * np.exp(d[k] * tau)))
        f[k] = np.exp(C[k] + D[k] * V + phy * np.log(S) * complex(0,1))
    return f[j]

