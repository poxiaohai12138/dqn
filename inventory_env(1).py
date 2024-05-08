#!/usr/bin/env python
# coding: utf-8

# In[1]:


# import numpy as np
# import pandas as pd
# 定义库存管理模型环境，按照GYM的格式
import gym
from gym import spaces
# from gym import logger, Env
# from gym.utils import seeding
import copy
import numpy as np
import random


# In[2]:


from gym.spaces import Box, Discrete, Dict, Tuple, MultiBinary, MultiDiscrete


# In[10]:


# low = np.array(
#     [
#         0,
# #                 到货量
#         0,
# #                 现有
#     ],
#     dtype=int,
# )
    
# high = np.array(
#     [
#         1000,
# #                 到货量
#         1000,
# #                 现有
#     ],
#     dtype=int,
# )
# #         self.action_space = spaces.Discrete(2)
# action_space =  Tuple((Discrete(1000), Box(0, 1, shape = (1,),dtype = float)))
# observation_space = spaces.Box(low, high,dtype = int)

# observation = observation_space.sample()
# observation[1]

#         action = action_space.sample()
#         action[0]


# In[11]:


from scipy import stats
# 设置random_state时，每次生成的随机数一样--任意数字
#不设置或为None时，多次生成的随机数不一样
# sample1 = stats.poisson.rvs(mu=8, size=14, random_state=None)
# sample2 = stats.poisson.rvs(mu=8)
# print(sample2)


# In[13]:


# 先进先出并设置腐烂成本 仓库内的生鲜必须在今天或明天卖出 如果今天的生鲜不能全部卖出则全部出库并计算费用
class InventoryEnv(gym.Env):
    def __init__(self):
#         0表示今天就过期 1表示明天过期,0表示今天的到货量，1表示库存量
        self.observation_space = spaces.Box(low=0, high=10000, shape=(2,), dtype=int)
        self.action_space = spaces.Discrete(100)
#         self.life_cycle = 3
#         self.lead_time = 1
        self.price = 1.2
        self.expired_cost = 1.2
        self.shortage_cost = 0.3
        self.current_state = [10,20]  
        self.quantity = self.current_state[0] + self.current_state[1]
        
    def reset(self):
        self.current_state = [random.randint(1, 100),random.randint(1, 100)]
        self.quantity = self.current_state[0] + self.current_state[1]
        self.demand = int(stats.poisson.rvs(mu=30))
        info = {'quantity': 0}
        return self.current_state,info,self.demand
                                       
    def render(self):
        pass

    def step(self,action):
        shortage_cost = 0
        expired_cost = 0
        new_state = copy.deepcopy(self.current_state)
        # generate demand
        self.demand = int(stats.poisson.rvs(mu=30))
        self.quantity = self.current_state[0] + self.current_state[1]

        new_state[0] = action
        if self.current_state[1] >= self.demand:
            new_state[1] = self.current_state[0]
            expired_quantity = self.current_state[1] - self.demand
            shortage_quantity = 0
        elif self.current_state[1] < self.demand and self.quantity >= self.demand:
            new_state[1] = self.quantity - self.demand
            expired_quantity, shortage_quantity = 0, 0
        elif self.quantity < self.demand:
            new_state[1] = 0
            expired_quantity = 0
            shortage_quantity = self.demand - self.quantity
            
        self.current_state = new_state
        expired_cost = expired_quantity*self.expired_cost
        shortage_cost = shortage_quantity*self.shortage_cost
        reward = -shortage_cost - expired_cost
        
        done = False
        info = {'quantity': self.quantity, 'demand': self.demand}
        return self.current_state,reward,done,info

