# -*- coding: utf-8 -*-


import os
import json
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
    parser.add_argument('-seed', type = int, help = 'rand seed', default = 19960207)
    parser.add_argument('-util', type = str, help = 'utility function', default = 'exp')
    parser.add_argument('-trancost', '-tc', type = str, help = 'transaction cost function', default = 'constant')
    parser.add_argument('-load', type = str, help = 'loading model')
    parser.add_argument('-nofit', action = 'store_true', help = 'bypass fitting')
    parser.add_argument('-suffix', '-s', type = str, help = 'suffix for dumping', default = rnd_suffix)
    parser.add_argument('-env', '-simdata', type = str, help = 'environment type', default = 'newgbm')
    parser.add_argument('-bs', action = 'store_true', help = 'bs policy lead')
    parser.add_argument('-qlbs_reward', '-qlbsr', action = 'store_true', help = 'reward type')
    parser.add_argument('-nosave', action = 'store_true', help = 'save model and out')
    args = parser.parse_args()

    print('-' * 10 + 'Run Args' + '-' * 10)

    for k, v in vars(args).items():
        print(f'{k}: {v}')

    np.random.seed(int(args.seed))


    # Select utility function
    
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
    
    if args.env == 'gbm':
        train_env = envs.GbmEnv(utility = utility, trancost = trancost)
        test_env = envs.GbmEnv(utility = utility, trancost = trancost)
    elif args.env == 'heston':
        train_env = envs.HestonEnv(utility = utility, trancost = trancost)
        test_env = envs.HestonEnv(utility = utility, trancost = trancost)
    elif args.env == 'newgbm':
        train_env = envs.NewEnv(utility = utility, trancost = trancost)
        test_env = envs.NewEnv(utility = utility, trancost = trancost)
    elif args.env == 'newheston':
        train_env = envs.NewHestonEnv(utility = utility, trancost = trancost)
        test_env = envs.NewHestonEnv(utility = utility, trancost = trancost)
    else:
        raise ValueError(f'Unknown environment: {args.env}')

    
    # Select network
    if 'new' in args.env:
        policy_net = models.NewPolicyModel()    # type: torch.nn.Module
        value_net = models.ValueModel()
    else:
        policy_net = models.PolicyModel()
        value_net = models.ValueModel()
        
    
    
    # Set up pipeline
    if 'new' in args.env:
        pipe = pipeline.NewPipe(policy_net = policy_net,
                                value_net = value_net,
                                train_env = train_env,
                                test_env = test_env,
                                bs_lead = args.bs,
                                qlbs_reward = args.qlbs_reward)         # type: pipeline.Pipeline
    else:
        pipe = pipeline.Pipeline(policy_net = policy_net,
                                 value_net = value_net,
                                 train_env = train_env,
                                 test_env = test_env,
                                 bs_lead = args.bs,
                                 qlbs_reward = args.qlbs_reward)
    
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
    
    if 'new' in args.env:
        rewards_test, final_wealth, actions_dql, cum_rewards = pipe.test(N = args.ten)
        delta_dql = torch.cumsum(torch.Tensor(actions_dql), dim = -1)
       
    else:
        rewards_test, final_wealth, delta_dql, cum_rewards = pipe.test(N = args.ten)
        actions_dql = torch.Tensor(delta_dql)[:, 1:] - torch.Tensor(delta_dql)[:, :-1]
    

    time_end = time.time()
    print('time cost', time_end - time_start, 's')

    # ---------------------------- 
    # Compute BS
    # ----------------------------

    if 'new' in args.env:
        delta_out = False
    else:
        delta_out = True

    test_data = np.array(pipe.test_data).swapaxes(-1, -2)
    test_Z = test_data[:, :, 1]
    test_pv = np.array(pipe.test_pv)         
    Reward_BS, actions_bs, delta_bs, pv_bs, cum_rd_bs = utils.reward_bs.re_bs(data = test_data, meta = pipe.test_env.sim_params, 
                                                                              utility = utility, trancost = trancost, qlbs_reward = args.qlbs_reward, delta_out = delta_out)        
    

    Reward_BS_dis, actions_bs_dis, delta_bs_dis, pv_bs_dis, cum_rd_bs_dis = utils.reward_bs.re_bs(data = test_data, meta = pipe.test_env.sim_params, 
                                                                                                  grids = pipe.all_actions().numpy(), 
                                                                                                  utility = utility, trancost = trancost, qlbs_reward = args.qlbs_reward, delta_out = delta_out)

    

    
   


    


