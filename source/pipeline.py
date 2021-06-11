# -*- coding: utf-8 -*-

# System
import os
from logging import getLogger
import time

# Computation
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
import torch
from torch import nn
# from torch._C import dtype, int16
from torch.optim import Adam
from torch.optim import SGD
from torch.optim import RMSprop
from torch.utils.data import WeightedRandomSampler

# Relatives
from .envs import GbmEnv
from .envs import NewEnv
from .envs import RealEnv
from .models import PolicyModel, RealPolicyModel
from .models import ValueModel



class Pipeline(object):
    
    
    def default_params(self):
        
        return {'batch_size'  : 1000,
                'lr_p'        : 1.e-7,
                'lr_v'        : 1.e-4,
                'replace_iter': 20,
                'memory_size' : 30000,
                'epsilon'     : 0.5}


    def pre_train(self, **kwargs):
        pass
    

    def __init__(self, policy_net, value_net, train_env, test_env, bs_lead = False, qlbs_reward = True, optimizer_p = SGD, optimizer_v = SGD, **kwargs):
        
        params = self.default_params()
        keys = params.keys()
        params.update(kwargs)
        params.update({'gamma': np.exp(- train_env.params['r'] * train_env.params['dt'])})
        
        self.params = {k: v for k, v in params.items() if k in keys}
        
        self.train_env = train_env
        self.test_env = test_env
        
        assert isinstance(train_env, GbmEnv)
        assert isinstance(test_env, GbmEnv)
        
        self.logger = getLogger('main')
        
        self.policy_net = policy_net
        self.value_net = value_net
        
        assert isinstance(self.policy_net, PolicyModel) or isinstance(self.policy_net, RealPolicyModel)
        assert isinstance(self.value_net, ValueModel)
        
        
        self.optimizer_p = optimizer_p(self.policy_net._model.parameters(),
                                       lr = self.params['lr_p'])
        self.optimizer_v = optimizer_v(self.value_net._model.parameters(),
                                       lr = self.params['lr_v'])

        self.optce = torch.optim.Adam(self.policy_net.parameters(), lr = 1e-6)
        
        self.bs_lead = bs_lead
        self.qlbs_reward = qlbs_reward

        self.reserve = {'mean_test_reward': [],
                        'mean_test_wealth': [],
                        'G': [],
                        'V': [],
                        'L': []}

    
    def get_path(self, delta_out = True, temp = 1.):
        
        """
        get 1 path of state, action and cumulative reward.
        Note: state_path doesn't contain stock[-1]. len(state_path) = 240, len(stock) = 241.
        coz t=0, position = 0.
        """
        
        stock, option = self.train_env.sim_data()
        s, done = self.train_env.reset(state_ex = (stock[0], option[0]))
        
        if self.bs_lead:
            a = self.bs_delta(stock[0], delta_out = delta_out)
            ai = np.argmin(abs(a - self.all_actions()))
            a = self.all_actions()[ai]
        else:
            with torch.no_grad():
                prob_a = self.policy_net._model.forward(torch.Tensor(s))
            a, ai = self.sample_action(prob_a, temp = temp)
        
        for S,Z in zip(stock[1:], option[1:]):
            
            s_, r, done_ = self.train_env.step(action = a, state_ex = (S, Z), qlbs_reward = self.qlbs_reward)
            
            self.train_env.store_path(s, a, ai, r)

            if done_:
                break
            
            s = self.train_env.state
            
            if self.bs_lead:
                a = self.bs_delta(S, delta_out = delta_out)
                ai = np.argmin(abs(a - self.all_actions()))
                a = self.all_actions()[ai]
            else:
                with torch.no_grad():
                    prob_a = self.policy_net._model.forward(torch.Tensor(s))
                a, ai = self.sample_action(prob_a, temp = temp)

        state_path = self.train_env.state_path 
        action_path = self.train_env.action_path
        actioni_path = self.train_env.action_index_path
        reward_path = self.train_env.reward_path

        G_path = self.gamma_mat().dot(reward_path)
        self.train_env.clear()

        return np.squeeze(state_path), np.squeeze(action_path), np.squeeze(actioni_path), np.squeeze(G_path)


    def gamma_mat(self):

        if hasattr(self, 'gammat'):

            return self.gammat

        else:
            
            T = self.train_env.sim_params['T']
            
            gm = np.eye(T)

            for t in range(1, T):

                gm += np.diag((self.params['gamma'] ** t) * np.ones(T - t), k = t)

            self.gammat = gm
            
            return gm

    def all_actions(self, lower_bound = 0, upper_bound = 1):
        """
        all actions in fixed range.
        """
        n_actions = self.policy_net.n_actions
        all_actions = torch.linspace(lower_bound, upper_bound, n_actions)
        return torch.squeeze(all_actions)

    def sample_action(self, prob_a, temp = None):
        """
        sample action based on the probability given by policy network.
        """
        assert all(prob_a >= 0), prob_a
        
        if temp is None:
            action_index = torch.argmax(prob_a)
        else:
            prob_a = self.warmer(prob_a, temperature = temp)
            action_index = list(WeightedRandomSampler(prob_a, 1))
        all_actions = self.all_actions()
        action = all_actions[action_index]
        return action, action_index

    
    def train(self, n_epoch = 10, delta_out = True, temp = 1., **kwargs):
        
        print('-' * 10 + 'Train' + '-' * 10)
        

        if torch.cuda.is_available():
            self.policy_net.cuda()
            self.value_net.cuda()
        
        for epoch in range(1, n_epoch + 1):
            epoch_loss = []
            loss = None
            state_path, action_path, action_index_path, G_path = self.get_path(delta_out = delta_out, temp = temp)
            delta_vs = []

            for t, (s, a, ai, G) in enumerate(zip(state_path, action_path, action_index_path, G_path)):
                
                if torch.cuda.is_available():
                    pass

                if t == 0:
                    self.reserve['G'].append(G)
                    with torch.no_grad():
                        v = self.value_net._model.forward(torch.Tensor(s))
                    self.reserve['V'].append(v.item())
                    self.reserve['L'].append(0.)
                
                with torch.no_grad():    
                    delta_v = G - self.value_net._model.forward(torch.Tensor(s))    
                    delta_vs.append(delta_v.item())
                
                g = torch.Tensor([self.params['gamma'] ** t])
                
                self.optimizer_v.zero_grad()
                
                lv = - delta_v * self.value_net._model.forward(torch.Tensor(s))
                
                lv.backward()
                self.optimizer_v.step()
                
                self.optimizer_p.zero_grad()
                
                pi_a = torch.squeeze(nn.functional.one_hot(
                    torch.Tensor([ai]).to(int),
                    num_classes = self.policy_net.n_actions
                ))
                
                logpi = torch.log(self.policy_net.forward(torch.Tensor(s)))

                assert pi_a.shape == logpi.shape[- len(pi_a.shape):], (pi_a.shape, logpi.shape)
                
                lp = - g * delta_v * (pi_a * logpi).sum()

                lp.backward()
                self.optimizer_p.step()
                
                with torch.no_grad():
                    v = self.value_net._model.forward(torch.Tensor(s))

                self.reserve['L'][-1] += lp.item()
            
            print(f"V {self.reserve['V'][-1]:.4f}, LP {self.reserve['L'][-1]:.8f}")
            
        return None
    
    
    def learn(self, **kwargs):
        raise self.train(**kwargs)
    
    
    def test(self, N):
        
        """
        Evaluation on test data.
        Calculate accumulated rewards with current target_net.
        """
        
        print('-' * 10 + 'Test' + '-' * 10)
        
        self.test_data = []
        self.test_pv = []
        self.rewards_test = []
        self.final_wealth = []
        self.cum_rewards = []
        
        actions = []
        discount_path = self.params['gamma'] ** np.arange(0, self.test_env.sim_params['T'] + 1)

        for n in range(N):
            cum_reward_episode = [0.]
            reward_episode = 0.
            action_episode = []
            pv_episode = []
            stock, option = self.test_env.sim_data()
            self.test_data.append([stock, option])  # N * 2 * T
            s, done = self.test_env.reset(state_ex = (stock[0], option[0]))
            pv_episode.append(self.test_env.portfolio_value())
            for S, Z in zip(stock[1:], option[1:]):
                s = self.test_env.state
                with torch.no_grad():
                    prob_a = self.policy_net._model.forward(torch.Tensor(s))
                a, _ = self.sample_action(prob_a, temp = None)
                s_, r, done = self.test_env.step(action = a, state_ex = (S, Z), qlbs_reward = self.qlbs_reward)
                pv_episode.append(self.test_env.portfolio_value())
                reward_episode += r
                cum_reward_episode.append(r.numpy())
                action_episode.append(a)
            cum_reward_episode = np.array(cum_reward_episode) * discount_path
            cum_reward_episode = np.cumsum(cum_reward_episode, axis = 0)
            wealth_episode = self.test_env.wealth()
            self.rewards_test.append(reward_episode)
            self.cum_rewards.append(cum_reward_episode)
            self.final_wealth.append(wealth_episode)
            self.test_pv.append(pv_episode)

            actions.append(action_episode)
        
            
        print(f"Mean Test Reward:\t{torch.mean(torch.Tensor(self.rewards_test)):.4f}.")
        print(f"Std Dev Test Reward:\t{torch.std(torch.Tensor(self.rewards_test)):.4f}.")
        print(f"Mean Final Wealth:\t{torch.mean(torch.Tensor(self.final_wealth)):.4f}.")
        print(f"Std Dev Final Wealth:\t{torch.std(torch.Tensor(self.final_wealth)):.4f}.")

        self.reserve['mean_test_reward'].append(torch.mean(torch.Tensor(self.rewards_test)))
        self.reserve['mean_test_wealth'].append(torch.mean(torch.Tensor(self.final_wealth)))
        
                    
        return torch.Tensor(self.rewards_test), torch.Tensor(self.final_wealth), actions, np.squeeze(self.cum_rewards)
    
    @staticmethod
    def warmer(prob_a, temperature):
        logp = torch.log(prob_a)
        prop = torch.exp(logp / temperature)
        return prop / torch.sum(prop)
    @staticmethod
    def plot_hist_final_wealth(wealth, baseline, suffix = None):
        
        suffix = str(np.random.uniform())[-6:] if suffix is None else suffix
        
        plt.hist((baseline, wealth), label = ('Black Scholes', 'DQN'))
        plt.legend(loc = 'upper left')
        plt.savefig(f'out/hist_fix_a_t-{suffix}.png')

        return None
    
    
    def state_dict(self):
        
        return {'policy': self.policy_net.state_dict(),
                'value': self.value_net.state_dict()}
        
        
    def save_state_dict(self, file):
        return torch.save(self.state_dict(), file)
    
        
    def load_state_dict(self, policy, value):
        
        self.policy_net.load_state_dict(policy)
        self.value_net.load_state_dict(value)
        
        return None

    def perturb_policy(self, sigma = 1.):
        pass

    def perturb_check(self, threshold):
        pass

    def bs_delta(self, stock, delta_out = True):
        tau = self.train_env.state[2]
        assert tau > 0
        K = self.train_env.sim_params['K']
        r = self.train_env.sim_params['r']
        sigma = self.train_env.sim_params['sigma']
        logSK = np.log(stock) - np.log(K)
        d1 = (logSK + (r + sigma ** 2 / 2) * tau) / (np.sqrt(tau) * sigma)
        if delta_out:
            return norm.cdf(d1) 
        else:
            return norm.cdf(d1) - self.train_env.state[1]

class NewPipe(Pipeline):
    """
    adapted to NewEnv.
    """
    def __init__(self, policy_net, value_net, train_env, test_env, bs_lead = False, qlbs_reward = True, optimizer_p = SGD, optimizer_v = SGD, **kwargs):
        super().__init__(policy_net = policy_net, value_net = value_net, train_env = train_env, test_env = test_env, bs_lead = bs_lead, qlbs_reward = qlbs_reward, optimizer_p = optimizer_p, optimizer_v = optimizer_v, **kwargs)
        assert isinstance(train_env, NewEnv)
        assert isinstance(test_env, NewEnv)

    def all_actions(self, lower_bound = -0.05, upper_bound = 0.05):
        """
        only 3 choices
        """
        assert self.policy_net.n_actions == 3, 'policy network output dimension not match'
        return torch.Tensor([lower_bound, 0., upper_bound])
        
    
    def train(self, n_epoch = 10, delta_out = False, temp = 1., **kwargs):
        return super().train(n_epoch, delta_out, temp, **kwargs)
    
    
    def pre_train(self, n_epoch, delta_out = True):

        cel = nn.CrossEntropyLoss()

        bslead = self.bs_lead

        self.bs_lead = True
        

        for epoch in range(1, n_epoch + 1):

            self.optce.zero_grad()

            state_path, action_path, action_index_path, G_path = self.get_path(delta_out = False)

            all_actions = np.unique(action_path)
            np.sort(all_actions)
            
            mapp = {a: i for i, a in enumerate(all_actions)}

            y = torch.Tensor([mapp[a] for a in action_path]).to(torch.long)

            policy_action = self.policy_net.forward(torch.Tensor(state_path))

            l = cel(policy_action, y)
            l.backward()

            self.optce.step()

            print(f'Ep: {epoch:02d}, CEL: {l.item()}')
        
        return None

class RealPipe(NewPipe):
    
    """adapted to RealEnv."""

    def __init__(self, policy_net, value_net, train_env, test_env, bs_lead = False, qlbs_reward = True, optimizer_p = SGD, optimizer_v = SGD, **kwargs):
        super().__init__(policy_net = policy_net, value_net = value_net, train_env = train_env, test_env = test_env, bs_lead = bs_lead, qlbs_reward = qlbs_reward, optimizer_p = optimizer_p, optimizer_v = optimizer_v, **kwargs)
        assert isinstance(train_env, RealEnv)
        assert isinstance(test_env, RealEnv)
    
    def gamma_mat(self, T):
        """adapted to different path length."""

        gm = np.eye(T)

        for t in range(1, T):

            gm += np.diag((self.params['gamma'] ** t) * np.ones(T - t), k = t)

        self.gammat = gm
            
        return gm


    def get_path(self, delta_out = False, temp = 1.):

        stock, option, taus, dones = self.train_env.sim_data()

        # calculate sigma for BS
        dt = self.train_env.sim_params['dt']
        y = np.log(stock)[1:] - np.log(stock)[:-1]
        bs_sigma = np.std(y) / np.sqrt(dt)

        s, done = self.train_env.reset(state_ex = (stock[0], option[0], taus[0], dones[0]))
        
        if self.bs_lead:
            a = self.bs_delta(stock[0], taus[0], bs_sigma, delta_out = delta_out)
            ai = np.argmin(abs(a - self.all_actions()))
            a = self.all_actions()[ai]
            print(a)
        else:
            with torch.no_grad():
                prob_a = self.policy_net._model.forward(torch.Tensor(s))
            a, ai = self.sample_action(prob_a, temp = temp)
        
        for S, Z, Tau, Done in zip(stock[1:], option[1:], taus[1:], dones[1:]):
            
            s_, r, done_ = self.train_env.step(action = a, state_ex = (S, Z, Tau, Done), qlbs_reward = self.qlbs_reward)
            
            self.train_env.store_path(s, a, ai, r)

            if done_:
                break

            
            s = self.train_env.state
            
            if self.bs_lead:
                a = self.bs_delta(S, Tau, bs_sigma, delta_out = delta_out)
                ai = np.argmin(abs(a - self.all_actions()))
                a = self.all_actions()[ai]
            else:
                with torch.no_grad():
                    prob_a = self.policy_net._model.forward(torch.Tensor(s))
                a, ai = self.sample_action(prob_a, temp = temp)

        
        state_path = self.train_env.state_path 
        action_path = self.train_env.action_path
        actioni_path = self.train_env.action_index_path
        reward_path = self.train_env.reward_path

        G_path = self.gamma_mat(T = len(reward_path)).dot(reward_path)

        self.train_env.clear()

        return np.squeeze(state_path), np.squeeze(action_path), np.squeeze(actioni_path), np.squeeze(G_path)

    def all_actions(self, lower_bound = -0.05, upper_bound = 0.05):
        return super().all_actions(lower_bound = lower_bound, upper_bound = upper_bound)

    def test(self, N):
        """
        Evaluation on test data.
        Calculate accumulated rewards with current target_net.
        """
        
        print('-' * 10 + 'Test' + '-' * 10)
        
        self.test_data = []
        self.test_pv = []
        self.rewards_test = []
        self.final_wealth = []
        self.cum_rewards = []
        
        actions = []
        
        for n in range(N):
            cum_reward_episode = [0.]
            reward_episode = 0.
            action_episode = []
            pv_episode = []
            while True:
                stock, option, taus, dones = self.test_env.sim_data()
                if option.mean() > 100:
                    break
            self.test_data.append([stock, option, taus, dones])  # N * 4 * T
            s, done = self.test_env.reset(state_ex = (stock[0], option[0], taus[0], dones[0]))
            pv_episode.append(self.test_env.portfolio_value())
            for S, Z, Tau, Done in zip(stock[1:], option[1:], taus[1:], dones[1:]):
                s = self.test_env.state
                with torch.no_grad():
                    prob_a = self.policy_net._model.forward(torch.Tensor(s))
                a, _ = self.sample_action(prob_a, temp = None)
                s_, r, done = self.test_env.step(action = a, state_ex = (S, Z, Tau, Done), qlbs_reward = self.qlbs_reward)
                pv_episode.append(self.test_env.portfolio_value())
                reward_episode += r
                cum_reward_episode.append(r.numpy())
                action_episode.append(a)
            discount_path = self.params['gamma'] ** np.arange(0, len(cum_reward_episode))
            cum_reward_episode = np.array(cum_reward_episode) * discount_path
            cum_reward_episode = np.cumsum(cum_reward_episode, axis = 0)
            wealth_episode = self.test_env.wealth()
            self.rewards_test.append(reward_episode)
            self.cum_rewards.append(cum_reward_episode)
            self.final_wealth.append(wealth_episode)
            self.test_pv.append(pv_episode)

            actions.append(action_episode)
            
        print(f"Mean Test Reward:\t{torch.mean(torch.Tensor(self.rewards_test)):.4f}.")
        print(f"Mean Final Wealth:\t{torch.mean(torch.Tensor(self.final_wealth)):.4f}.")
        print(f"Std Dev Final Wealth:\t{torch.std(torch.Tensor(self.final_wealth)):.4f}.")

        self.reserve['mean_test_reward'].append(torch.mean(torch.Tensor(self.rewards_test)))
        self.reserve['mean_test_wealth'].append(torch.mean(torch.Tensor(self.final_wealth)))
        
                    
        return torch.Tensor(self.rewards_test), torch.Tensor(self.final_wealth), actions, self.cum_rewards


    def bs_delta(self, stock, tau, sigma, delta_out):
        """path-mode"""

        tau_p = tau[tau != 0]
        tau_0 = tau[tau == 0]
        stock_p = stock[tau != 0]
        stock_0 = stock[tau == 0]

        deltas = np.zeros_like(stock)

        K = self.train_env.sim_params['K']
        r = self.train_env.sim_params['r']
        logsk = np.log(stock_p) - np.log(K)
        d1 = (logsk + (r + sigma ** 2 / 2)* tau_p) / (sigma * np.sqrt(tau_p))
        delta = norm.cdf(d1)
        deltas[tau != 0] = delta
        deltas[tau == 0] = np.where(stock_0 > K, 1., 0.)

        if delta_out is None:
            print('only used in final BS test.')
            return deltas - np.append(0., deltas[:-1]), deltas
        elif delta_out:
            return deltas
        else:
            return deltas - self.train_env.state[1]
        
    
    