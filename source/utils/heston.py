__all__ = ['Data_heston_test']
import numpy as np 

from ..utils import Heston_price


def data_heston(N, T, mu = 0.1 / 365, kappa = 0.02, theta = 0.01, sigma = 0.001, rho = 0.5, S0 = 1., V0 = 0.01, dt = 1.):
    '''
    N paths, T steps.
    '''
    logprice = [np.repeat(np.log(S0), N)]
    V = [np.repeat(V0, N)]
    for t in range(1, T+1):
        W_dt = np.random.multivariate_normal([0,0], [[dt, rho * dt], [rho * dt, dt]], N)
        W_dt_s = W_dt[:, 0]
        W_dt_v = W_dt[:, 1]
        delta_V = kappa * (theta - np.array(V)[-1, :]) * dt + sigma * np.sqrt(np.array(V)[-1, :]) * W_dt_v
        V = np.append(V, [np.array(V)[-1, :] + delta_V], axis = 0)
        
        assert np.array(V).all() > 0, 'negative volatility occurs'
        delta_logS = mu * dt + np.sqrt(np.array(V)[-1, :]) * W_dt_s
        logprice = np.append(logprice, [np.array(logprice)[-1, :] + delta_logS], axis = 0)
    

    return np.squeeze(np.array(logprice).T), np.squeeze(np.array(V).T)   

def data_heston_op_price(kappa, theta, sigma, rho, S, V, K, r, dt = 1.):
    N, T = S.shape

    tau = np.arange(T, 0, -1)

    htprice = []

    for n in range(N):
        ht = Heston_price(
            kappa = kappa, theta = theta, sigma = sigma, rho = rho, 
            S = S[n, :-1], V = V[n, :-1], K = K, r = r, tau = tau[:-1], dt = dt
            )
        htprice.append(
            np.append(ht, max(S[n, -1] - K, 0))
        )
        

    return np.squeeze(np.array(htprice))


def Data_heston_test(N):

    mu = 0.1 / 365
    sigma = 0.01 / np.sqrt(365)
    kappa = 0.5 / 365
    theta = 0.05 / 365
    rho = -0.07
    r = 0.03 / 365
    S0 = 100
    V0 = 0.002
    K = 100
    T = 240
    dt = 1.

    logprice, V = data_heston(N = N, T = T, 
                              mu = mu, kappa = kappa, theta = theta, sigma = sigma, rho = rho, 
                              S0 = S0, V0 = V0, dt = dt)
    price = np.exp(logprice)
    if N == 1:
        price = np.reshape(price, [1, price.shape[-1]]) 
        V = np.reshape(V, [1, V.shape[-1]])
    else:
        pass
    heston_op = data_heston_op_price(
        kappa = kappa, theta = theta, sigma = sigma, rho = rho, S = price, V = V, K = K, r = r, dt = dt
        )

    meta = {'N': N, 'S0': S0, 'K': K, 'T': T, 'mu': mu, 'sigma': sigma, 'dt': dt, 'r': r}
    return (price, heston_op, V), meta

