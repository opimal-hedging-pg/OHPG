# -*- coding: utf-8 -*-



import numpy as np
from scipy.stats import norm
import torch
import os
import json
import random
import pandas as pd
import matplotlib.pyplot as plt

import torch
from .util_funcs import exp_utility
from .trancost_funcs import constant_trancost
    
class GbmEnv(object):
    
    """
    simulate gbm data in environment.
    """
    
    
    def __init__(self, utility = None, trancost = None, **kwargs):
        
        # Utility function
        
        if callable(utility):
            self.utility = utility
        else:
            self.utility = exp_utility(risk_pref = 0.)
        
        # Transaction cost function

        if callable(trancost):
            self.trancost = trancost
        else:
            self.trancost = constant_trancost(tc_para = 0.)

        
        # Parameters
        
        params = self.default_params()
        keys = params.keys()
        params.update(kwargs)
        
        self.params = {
            k: v for k, v in params.items() if k in keys
        }

        params_sim = self.default_sim_params()
        keys_sim = params_sim.keys()
        params_sim.update(kwargs)
        
        
        self.sim_params = {
            k: v for k, v in params_sim.items() if k in keys_sim
        }
        
        
    def default_params(self):
        """
        Only relevant to accumulating bank account.
        TODO: T should be determined in data sequences. Not here.
        """
        return {'r': .03 / 365, 'dt': 1., 'T': 240}
        

    def default_sim_params(self):
        """
        Parameters for simulating data
        """
        return {'mu': 0.1 / 365,
                'r': .03 / 365,
                'sigma': .1 / np.sqrt(365),
                'dt': 1., 'S0': 100, 'K': 100, 'T': 240}
    
    def sim_data(self):
        """
        simulate a sample path of stock price and option price.
        """
        mu = self.sim_params['mu']
        r = self.sim_params['r']
     
        sigma = self.sim_params['sigma']
        dt = self.sim_params['dt']
    
        S0 = self.sim_params['S0']
        K = self.sim_params['K']
        T = self.sim_params['T']

        drifts = mu * np.arange(1, T + 1) * dt
        logS0 = np.log(S0)
        diffusion = sigma * np.random.normal(0, np.sqrt(dt), T).cumsum()
        logprice = logS0 + drifts + diffusion
        logprice = np.append(logS0, logprice)
        
        price = np.exp(logprice)

        tau = np.arange(T, 0, -1)
        logsk = np.log(price[:-1]) - np.log(K)
        
        d1 = (logsk + (r + 0.5 * sigma ** 2) * tau) / (sigma * np.sqrt(tau))
        d2 = d1 - sigma * np.sqrt(tau)
        bs_p = norm.cdf(d1) * price[:-1] - norm.cdf(d2) * K * np.exp(- r * tau)
        bs_price = np.append(bs_p, max(price[-1] - K, 0))
        
        return np.squeeze(price), np.squeeze(bs_price)  


    
    def wealth(self):
        return self.state[0] + self.state[1] * self.state[3] - self.state[4]

    def portfolio_value(self):
        """bank money and stock value."""
        return self.state[0] + self.state[1] * self.state[3]
        
            
    def reset(self, state_ex, B0 = None):
        
        """
        state_en:
            (back_account, position, time_to_go (step_counts))
        state_ex:
            (stock_price, option_price, ...)
        """
        
        S0, Z0 = state_ex
        
        B0 = Z0 if B0 is None else B0
        self.state = (B0, 0., self.params['T'], S0, Z0)
        self.done = False
        
        return (B0, 0., self.params['T'], S0, Z0), False

    
    def step(self, action, state_ex, qlbs_reward = True):
        
        """
        Suppose action is taken upon previous state.
        Calculate reward & next state.
        """
        
        B, delta, tau, S, Z = self.state
        term = self.done
        
        # Process internal states
        delta_ = action
        tau_ = tau - 1
        
        B_ = (B - (action - delta) * S - self.trancost((action - delta) * S)) * np.exp(self.params['r'] * self.params['dt'])
                
        # Process external states
        S_, Z_ = state_ex
        
        # Combine
        s_ = (B_, delta_, tau_, S_, Z_)
        term_ = (tau_ == 0)
        
        # Reward
        if qlbs_reward:
            deltap = delta_ * (S_ * np.exp(- self.params['r'] * self.params['dt']) - S) - self.trancost((delta_ - delta) * S)
        else:
            deltap = np.exp(- self.params['r'] * self.params['dt']) * B_ - B - (np.exp(- self.params['r'] * self.params['dt']) * Z_ - Z) + np.exp(- self.params['r'] * self.params['dt']) * delta_ * S_ - delta * S

        self.state = s_

        assert (not torch.isnan(deltap)), 'Delta P is NaN.'

        
        return s_, self.utility(deltap), term_

    def store_path(self, state, action, action_index, reward):
        """
        store transition path.
        """
        try:
            state_path = self.state_path
            action_path = self.action_path
            action_index_path = self.action_index_path
            reward_path = self.reward_path
        except:
            self.clear()
            
        self.state_path.append(state)
        self.action_path.append(action)
        self.action_index_path.append(action_index)
        self.reward_path.append(reward)

        return None


    def clear(self):

        self.state_path = []
        self.action_path = []
        self.action_index_path = []
        self.reward_path = []

        return None

class HestonEnv(GbmEnv):
    """
    load Heston data.
    """

    def __init__(self, utility = None, trancost = None, **kwargs):
        super().__init__(utility = utility, trancost = trancost, **kwargs)
    
    def default_sim_params(self):
        return {'T' : 240}

    def sim_data(self):


        path = './data/heston1/'
        dir = os.listdir(path)

        random.shuffle(dir)
        
        if '.json' not in dir[0]:
            lt = np.load(f'{path}{dir[0]}', allow_pickle = True)
        else:
            lt = np.load(f'{path}{dir[1]}', allow_pickle = True)
        
        price = lt[0]
        bs_price = lt[1]

        with open(f'{path}/meta.json', 'r') as f:
            self.meta = json.load(f)
            self.sim_params.update(self.meta)

        return np.squeeze(price), np.squeeze(bs_price)
    

class NewEnv(GbmEnv):
    """
    actions can be 0.1, -0.1 and 0.
    action is not position.
    """
    def __init__(self, utility = None, trancost = None, **kwargs):
        super().__init__(utility = utility, trancost = trancost, **kwargs)
    
    def step(self, action, state_ex, qlbs_reward = True):
        
        """
        Suppose action is taken upon previous state.
        Calculate reward & next state.
        """

        
        B, delta, tau, S, Z = self.state
        term = self.done
        
        # Process internal states
        delta_ = action + delta
        tau_ = tau - 1
        
        B_ = (B - action * S - self.trancost(action * S)) * np.exp(self.params['r'] * self.params['dt'])
                
        # Process external states
        S_, Z_ = state_ex
        
        # Combine
        s_ = (B_, delta_, tau_, S_, Z_)
        term_ = (tau_ == 0)
        
        # Reward
        if qlbs_reward:
            deltap = delta_ * (S_ * np.exp(- self.params['r'] * self.params['dt']) - S) - self.trancost(action * S)
        else:
            deltap = B_ - B - (Z_ - Z) + delta_ * S_ - delta * S

        self.state = s_

        assert (not torch.isnan(deltap)), 'Delta P is NaN.'

        
        return s_, self.utility(deltap), term_

class NewHestonEnv(HestonEnv, NewEnv):
    """
    Heston data in NewEnv step mode.
    """
    def __init__(self, utility = None, trancost = None, **kwargs):
        super().__init__(utility = utility, trancost = trancost, **kwargs)
        print(NewHestonEnv.__mro__)
        print(super())

    def default_sim_params(self):
        """ HestonEnv.default_sim_params() """
        return super().default_sim_params()

    def sim_data(self):
        """ HestonEnv.sim_data() """
        return super().sim_data()

    def step(self, action, state_ex, qlbs_reward = True):
        """NewEnv.step()"""
        return super(HestonEnv, self).step(action, state_ex, qlbs_reward)



class RealEnv(NewEnv):
    """adapted to real sp500 data."""

    def __init__(self, utility = None, trancost = None, **kwargs):
        super().__init__(utility = utility, trancost = trancost, **kwargs)
    
    def default_params(self):
        return {'r': .03 / 365, 'dt': 1.}

    def default_sim_params(self):
        return {'r': .03 / 365,
                'dt': 1., 
                'K': 1800}

    def sim_data(self):

        df = pd.read_csv('./data/sp500/strike1800.txt', sep = '\t')
        path_num = int((df.shape[-1] - 2) / 2)
        rnd_ind = random.randint(0, path_num - 1)
        lt = df[['sp500', f'tau-{rnd_ind}', f'op_price-{rnd_ind}']].dropna().values.tolist()
        stock = np.array(lt)[:, 0]
        tau = np.array(lt)[:, 1]
        op_price = np.array(lt)[:, 2]
        done = np.append(np.repeat(False, np.shape(tau)[0] - 1),True)
        assert len(done) == len(tau), 'length of done is wrong!'

        return np.squeeze(stock), np.squeeze(op_price), np.squeeze(tau), np.squeeze(done)
        
    
    def reset(self, state_ex, B0 = None):
        S0, Z0, tau0, term = state_ex
        B0 = Z0 if B0 is None else B0
        self.state = (B0, 0., tau0, S0, Z0)
        self.done = False

        return (B0, 0., tau0, S0, Z0), False

    def step(self, action, state_ex, qlbs_reward = True):
        """tau is external variable."""
        B, delta, tau, S, Z = self.state
        term = self.done
        
        # Process internal states
        delta_ = action + delta
        
        B_ = (B - action * S - self.trancost(action * S)) * np.exp(self.params['r'] * self.params['dt'])
                
        # Process external states
        S_, Z_, tau_, term_ = state_ex
        
        # Combine
        s_ = (B_, delta_, tau_, S_, Z_)

        
        # Reward
        if qlbs_reward:
            deltap = delta_ * (S_ * np.exp(- self.params['r'] * self.params['dt']) - S) - self.trancost(action * S)
        else:
            deltap = B_ - B - (Z_ - Z) + delta_ * S_ - delta * S

        self.state = s_

        assert (not torch.isnan(deltap)), 'Delta P is NaN.'
        
        return s_, self.utility(deltap), term_


    

