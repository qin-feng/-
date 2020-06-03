#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar  2 05:29:51 2020

@author: wmz

It is a main function used for UAV reforcement learning
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.nn.utils as utils
from torch.autograd import Variable
import cv2
import numpy as np


class Policy(nn.Module):
    #Define policy function for reforcement learning
    def __init__(self, hidden_size, num_inputs, action_space):
        super(Policy, self).__init__()
        self.action_space = action_space
        num_outputs = action_space
        #Two linear layers
        #For debug usage here
        #print("The Structure of the neural network is {} {} {}".format(num_inputs,hidden_size,action_space))
        #End of debug
        self.linear1 = nn.Linear(num_inputs, hidden_size)
        self.linear2 = nn.Linear(hidden_size, num_outputs)
        

    def forward(self, inputs):
        #Define for forward propagation
        x = inputs
        #For debug usage here
        print(np.shape(x))
        print(self.linear1)
        print(self.linear2)
        #End of debug
        x = F.relu(self.linear1(x))
        #print("The output of the first linear layer is: {}".format(np.shape(x)))
        action_scores = self.linear2(x)        
        #print("The output of the second linear layer is: {}".format(np.shape(action_scores)))
        return F.softmax(action_scores) 


class REINFORCE:
    #It is the reforcement learning policy define
    def __init__(self, hidden_size, num_inputs, action_space):
        #Define the policy for rl
        self.action_space = action_space
        self.model = Policy(hidden_size, num_inputs, action_space)
        self.optimizer = optim.Adam(self.model.parameters(), lr=1e-3)

    def select_action(self, state):
        #For debug usage
        #print(np.shape(state))
        #End of debug
        probs = self.model(Variable(state))
        action = probs.multinomial(2,replacement=True).data
        #print("The action is {}".format(action))
        prob = probs[action[0]].view(1,-1)
        log_prob = prob.log()
        entropy = - (probs*probs.log()).sum()
        #print(action)
        return action[0], log_prob, entropy

    def update_parameters(self, rewards, log_probs, entropies, gamma):
        R = torch.zeros(1, 1)
        loss = 0
        for i in reversed(range(len(rewards))):
            R = gamma * R + rewards[i]
            loss = loss - (log_probs[i]*(Variable(R).expand_as(log_probs[i]))).sum()
            #loss = loss - (log_probs[i]*(Variable(R).expand_as(log_probs[i]))).sum() - (0.0001*entropies[i]).sum()
        loss = loss / len(rewards)
        self.optimizer.zero_grad()
        loss.backward()
        utils.clip_grad_norm(self.model.parameters(), 40)
        self.optimizer.step()

class searchproject:
    #project used to define all functions used in this code
    def __init__(self, inputmap, stepth, scalefactor, respect=[1,1], chopsize=[0,0,0,0], num_episodes=500, num_steps=1000, gamma=0.9, err=0.01):
        #Parameter Define:
        #inputmap is the map that are used for test, it can be map with grey scale as pollution material density or otherwise
        #stepth is the size of the search step
        self.stepth = stepth
        #scalefactor is used as the scale factor for simulation
        #respect is the x axis or y axis ratio
        #chopsize is used to chop part of map for simulation
        #num_episodes are episodes for UAV Path Planning
        #num_steps is number of steps for UAV Path Planning
        #gamma and err are planning parameters
        #It is used to define the project (map and step)
        #inputmap is the str defined for the inputmap
        ####################Main body############
        inputmap = cv2.imread(inputmap)
        #Reshape this inputmap to our required size
        self.roadmap = self.rescaleimg(inputmap, scalefactor, respect, chopsize)
        self.roadmap=255-self.roadmap
        #DEBUG usage
        print("Out put the following parameters: shape of the map used for search, shape of the original map, the scale factor, the respect value and the size of chop \n")
        #End of debug code
        print(np.shape(self.roadmap),np.shape(inputmap),scalefactor,respect,chopsize)
        #Define the agent function
        self.agent = REINFORCE(30,1,4)
        #Define action parameters
        self.ACTION_UP = 0
        self.ACTION_DOWN = 1
        self.ACTION_LEFT = 2
        self.ACTION_RIGHT = 3
        #Define learning properties
        self.gamma = gamma
        self.err = err


    def rescaleimg(self,orgimg,scalefactor,respect=[1,1], chopsize=[0,0,0,0]):
        #To check whether we need to chop this image or rescale image
        orgimg = orgimg[:,:,0]
        if respect == [1,1] and chopsize == [0,0,0,0]:
            #Used for simply rescale
            #Defined to rescale the original map to smaller one
            width = int(orgimg.shape[1]*scalefactor/100)
            height = int(orgimg.shape[0]*scalefactor/100)
            dsize = (width,height)
            #outimg = cv2.resize(orgimg,dsize)
        elif respect != [1,1] and chopsize == [0,0,0,0]:
            #rescale the original images according to defined respect value
            #The respect value is used to make the original image to respect size
            width = int(orgimg.shape[1]*scalefactor/100/respect[1])
            height = int(orgimg.shape[0]*scalefactor/100/respect[0])
            dsize = (width,height)
            #outimg = cv2.resize(orgimg,dsize)
        elif respect != [1,1] and chopsize != [0,0,0,0]:
            orgimg = orgimg[chopsize[0]:chopsize[1],chopsize[2]:chopsize[3]]
            width = int(orgimg.shape[1]*scalefactor/100/respect[1])
            height = int(orgimg.shape[0]*scalefactor/100/respect[0])
            dsize = (width,height)
        else:
            orgimg = orgimg[chopsize[0]:chopsize[1],chopsize[2]:chopsize[3]]
            width = int(orgimg.shape[1]*scalefactor/100/respect[1])
            height = int(orgimg.shape[0]*scalefactor/100/respect[0])
            dsize = (width,height)
        #We output the outimg for our simulation
        outimg = cv2.resize(orgimg,dsize)
        return outimg

    def diff(self, site, new_site):
        #Used to calculate the gradient difference between two sites
        x0 = site[0]
        y0 = site[1]
        x1 = new_site[0]
        y1 = new_site[1]
        grad = (self.roadmap[x1][y1] - self.roadmap[x0][y0])
        print(grad)
        return grad

    def step(self, site, action, stepth):
        #It is used to calculate the reward of action
        i, j = site
        #Used to calculate reward of four directions
        if action == self.ACTION_UP:
            #I think there is problem here
            #should import steps here
            return [max(i - stepth, 0), j]
        elif action == self.ACTION_DOWN:
            return [min(i + stepth, 34), j]
        elif action == self.ACTION_LEFT:
            return [i, max(j - stepth, 0)]
        elif action == self.ACTION_RIGHT:
            return [i, min(j + stepth,34)]
        else:
            assert False

    def searchiterate(self,num_episodes,initpoint1=[20,25],initpoint2=[24,25]):
        #We use this function to search the possible way
        for i_episode in range(num_episodes):
            #TODO: should solve this initial points 
            site1 = initpoint1
            site2 = initpoint2
            #Define the state now
            state = self.diff(site2,site1)
            #Transform to tensor
            state = torch.Tensor([state])
            #state = torch.transpose(state,0,1)
            #For debug useage
            #print(state)
            #End of debug
            entropies = []
            log_probs = []
            rewards = []
            reward = 0
            i = 0
            while True:
                i += 1
                #For Debug usage
                print("In iteration {} and the shape of state is {} {}".format(i,np.shape(state)[0],np.shape(state)[1]))
                #End of debug here
                action, log_prob, entropy = self.agent.select_action(state)
                #Make motion along a direction
                new_site = self.step(site2, action, self.stepth)
                new_state = self.diff(new_site, site2)
                #
                reward = reward - 0.1*i
                #update new state and site here
                site2 = new_site
                state = new_state
                #Evaluate cross entropies here
                entropies.append(entropy)
                log_probs.append(log_prob)
                rewards.append(reward)
                state = torch.Tensor([new_state])
                #Evaluate the state values of the current state
                x = new_site[0]
                y = new_site[1]
                value = self.roadmap[x][y]
                #It is not correct to direct set maximal value here
                if value in [255,254,253,252,251,250,249,248,247,246,245,244]:
                    break
                if i > 1000:
                    break
            #Update parameters of the enforcement learning here
            self.agent.update_parameters(rewards, log_probs, entropies, self.gamma)
            #For debug useage here
            print(i,value)
            #Update the agent parameters and save the torch model
            if i_episode % 50 == 0:
                torch.save(agent.model.state_dict(), os.path.join(root_dir, 'reinforce-'+str(i_episode)+'.pkl'))
            print("Episode: {}, reward: {}".format(i_episode, np.sum(rewards)))
