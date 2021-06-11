# -*- coding: utf-8 -*-

import os
import json
from source.utils.bs import BS_delta
from typing import Type
from argparse import ArgumentParser

import numpy as np
import torch
import time
import random
import seaborn as sns 
from sklearn.metrics import accuracy_score



from source import models
from source import envs
from source import pipeline
from source import utils
from source import util_funcs
from source import trancost_funcs


import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

###################


if __name__ == '__main__':
    
    time_start = time.time()
    cuda = torch.device('cuda:0')
    
    rnd_suffix = str(np.random.uniform())[-8:]

    parser = ArgumentParser()

    parser.add_argument('-rp', type = float, help = 'risk preference', default = 0.)
    parser.add_argument('-tcpara', type = float, help = 'parameter of transaction cost function', default = 0.)
    parser.add_argument('-ten', type = int, help = 'test size', default = 200)
    parser.add_argument('-ep', type = int, help = 'epoch', default = 100)
    parser.add_argument('-ep2', type = int, help = 'epoch in each train call', default = 1)
    parser.add_argument('-util', type = str, help = 'utility function', default = 'exp')
    parser.add_argument('-trancost', '-tc', type = str, help = 'transaction cost function', default = 'constant')
    parser.add_argument('-load', type = str, help = 'loading model')
    parser.add_argument('-nofit', action = 'store_true', help = 'bypass fitting')
    parser.add_argument('-suffix', '-s', type = str, help = 'suffix for dumping', default = rnd_suffix)
    parser.add_argument('-bs', action = 'store_true', help = 'bs policy lead')
    parser.add_argument('-nosave', action = 'store_true', help = 'save model and out')
    args = parser.parse_args()

    print('-' * 10 + 'Run Args' + '-' * 10)

    for k, v in vars(args).items():
        print(f'{k}: {v}')
    
    if args.util == 'exp':
        utility = util_funcs.exp_utility(args.rp)          # type: util_funcs.BaseUtility
    elif args.util == 'crra':
        utility = util_funcs.crra_extended(args.rp)
    elif args.util == 'gau':
        utility = util_funcs.gaussian_utility(args.rp)
    elif args.util == 'abs':
        utility = util_funcs.absol_utility(args.rp)
    elif args.util == 'eps':
        utility = util_funcs.epsilon_utility(args.rp)
    elif args.util == 'meanvar':
        utility = util_funcs.mean_var_utility(args.rp)
    elif args.util == 'lingau':
        utility = util_funcs.lingau_utility(args.rp)
    elif args.util == 'quad':
        utility = util_funcs.quad_utility(args.rp)
    else:
        raise ValueError(f'Unknown utility: {args.util}')

    if args.trancost == 'constant':
        trancost = trancost_funcs.constant_trancost(args.tcpara)       # type: trancost_funcs.BaseTrancost
    elif args.trancost == 'prop':
        trancost = trancost_funcs.proportional_trancost(args.tcpara)
    else:
        raise ValueError(f'Unknown utility: {args.trancost}')
    
    # Select env

    train_env = envs.RealEnv(utility = utility, trancost = trancost)
    test_env = envs.RealEnv(utility = utility, trancost = trancost)

    
    # Select network
    
    policy_net = models.RealPolicyModel()
    value_net = models.ValueModel()    
    
    
    # Set up pipeline

    pipe = pipeline.RealPipe(policy_net = policy_net,
                             value_net = value_net,
                             train_env = train_env,
                             test_env = test_env,
                             bs_lead = args.bs,
                             qlbs_reward = True)
    # Dump info before fitting
    
    info = vars(args)
    
    info.update({'policymodelname': pipe.policy_net.name,
                 'valuemodelname': pipe.value_net.name,
                 'utilityname': pipe.train_env.utility.name})

    if args.nofit or args.nosave:
        pass
    else:
        if os.path.exists(f'save/{args.suffix}'):
            pass
        else:
            os.makedirs(f'save/{args.suffix}')

        if os.path.exists(f'out/{args.suffix}'):
            pass
        else:
            os.makedirs(f'out/{args.suffix}')
        
        with open(f'save/{args.suffix}/info.json', 'w') as _f:
            json.dump(info, _f, indent = 4)
            _f.close()
        
        print('Info dumped.')    
    
    # Load model, if any
    if args.load is not None:
        states = torch.load(args.load)
        pipe.load_state_dict(**states)

    else:
        pass
    
    # Loop
    
    if args.nofit:
        
        pass
    
    else:
    
        temp = 1.
        for epoch in range(1, args.ep + 1):
            
            pipe.train(n_epoch = args.ep2, temp = temp)
            
            if epoch % 10 == 0:
                
                # Dump
                fn = '-'.join([pipe.policy_net.name, pipe.value_net.name, str(args.suffix)])                
                pipe.save_state_dict(f'save/{args.suffix}/{fn}-ep{epoch:02d}.pt')
                print(f'Model state save to save/{args.suffix}/{fn}-ep{epoch:02d}.pt')


    
    # Finalize
    
    rewards_test, final_wealth, actions_dql, cum_rewards = pipe.test(N = args.ten)
    dql_delta = []
    dql_action = []
    for i in range(len(actions_dql)):
        action = actions_dql[i]
        delta = torch.cumsum(torch.Tensor(actions_dql[i]), dim = -1)
        dql_delta.append(delta)
        dql_action.append(torch.Tensor(actions_dql[i]))
    
    time_end = time.time()
    print('time cost', time_end - time_start, 's')

    # ---------------------------- 
    # Temp: Compute BS
    # ----------------------------
    
    delta_out = False

    assert len(pipe.test_data) == args.ten, 'storage of test data is wrong.'
    bs_actions = []
    bs_actions_dis = []
    bs_deltas = []
    bs_deltas_dis = []
    bs_rewards = []
    bs_finals = []
    bs_pvs = []
    cum_rd_bs_dis = []
    
    r = pipe.test_env.sim_params['r']
    dt = pipe.test_env.sim_params['dt']
    
    for i in range(len(pipe.test_data)):
        s, z, tau, done = pipe.test_data[i]
        bs_B = [z[0]]
        T = len(s)

        y = np.log(s)[1:] - np.log(s)[:-1]
        sigma = np.std(y) / np.sqrt(pipe.test_env.sim_params['dt'])

        bs_action, bs_delta = pipe.bs_delta(s, tau, sigma, delta_out = None)
        bs_action = bs_action[:-1]
        bs_delta = bs_delta[:-1]
        grid = pipe.all_actions().numpy()
        bs_action_dis = []
        for i in range(len(bs_action)):
            ind = np.argmin(abs(bs_action[i] - grid))
            bs_action_dis.append(grid[ind])
        bs_action_dis = np.array(bs_action_dis)
        bs_delta_dis = np.cumsum(bs_action_dis)
        
        for a,s_ in zip(bs_action_dis, s[:-1]):
            B_ep = (bs_B[-1] - a * s_) * np.exp(r * dt)
            bs_B.append(B_ep)
        bs_B = np.array(bs_B)
        bs_pv = bs_B + np.append(0., bs_delta_dis) * s

        bs_reward = bs_delta_dis * (s[1:] * np.exp(-r * dt) - s[:-1])
        gamma = pipe.params['gamma']
        discount = gamma ** np.arange(0, len(bs_reward), 1)
        bs_reward *= discount
        bs_cum_reward = np.append(0., np.cumsum(bs_reward))

        bs_final = z[0] * np.exp(r * T) + bs_delta_dis[-1] * s[-1] - np.sum(bs_action_dis * s[:-1] * np.exp(r * np.arange(T-1, 0, -1))) - z[-1]

        bs_actions.append(bs_action)
        bs_actions_dis.append(bs_action_dis)
        bs_deltas.append(bs_delta)
        bs_deltas_dis.append(bs_delta_dis)
        bs_rewards.append(np.sum(bs_reward))
        bs_finals.append(bs_final)
        bs_pvs.append(bs_pv)
        cum_rd_bs_dis.append(bs_cum_reward)
    
    


        
       
        




