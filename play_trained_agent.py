print('start')

import gym
import ptan
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn


env = gym.make('LunarLander-v2')

class PolicyNetwork(nn.Module):
    """
    PGN-Policy Gradient Network (Function Approximator):
    
    - input size: number of observation states in environment
    - n_actions: number of actions in environment
    
    We will use neural network with one hidden layer, which 
    has one hidden layer with 128 neurons and ReLU activation function, 
    as was described in report. So, our vector of \theta parameters 
    will have 128 elements. 
    """
    # initialization function
    def __init__(self, input_size, n_actions):
        super(PolicyNetwork, self).__init__()
        # Creating a simple neural network with 128 parameters \theta
        # in one hidden layer 
        # Softmax activation will be done in the algorithm later.
        self.net = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.ReLU(),
            nn.Linear(128, n_actions)
        )
    
    # forward propagation function
    def forward(self, x):
        return self.net(x)



net = PolicyNetwork(env.observation_space.shape[0], env.action_space.n)
    
net.load_state_dict(torch.load('lunar_lander_model.pt'))
    
agent = ptan.agent.PolicyAgent(net, preprocessor=ptan.agent.float32_preprocessor,
                                               apply_softmax=True)

env = gym.make('LunarLander-v2')
obs = env.reset()

while True:
    env.render()
    obs_v = torch.FloatTensor(obs)
    logits_v = net(obs_v)
    action = np.argmax(logits_v.data.cpu().numpy()) 
    obs, r, done, _ = env.step(action)
    if done:
        print('done')
        break
        
env.close()

print('end')
