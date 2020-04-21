import os
import numpy as np
import cv2
import torch
from PG1 import REINFORCE

num_episodes = 500
num_steps = 1000
gamma = 0.9
err = 0.01

image =cv2.imread('/home/wmz/Documents/1/kuosan1.png')
img = 255-image[:,:,1]
im = cv2.resize(img,(35,35))

ACTION_UP = 0
ACTION_DOWN = 1
ACTION_LEFT = 2
ACTION_RIGHT = 3
stepth = 4

def diff(site,new_site):
    x0 = site[0]
    y0 = site[1]
    x1 = new_site[0]
    y1 = new_site[1]
    diff = (im[x1][y1] - im[x0][y0])/10
    return diff

def step(site, action, stepth):
    i, j = site
    if action == ACTION_UP:
        return [max(i - stepth, 0), j]
    elif action == ACTION_DOWN:
        return [min(i + stepth, 34), j]
    elif action == ACTION_LEFT:
        return [i, max(j - stepth, 0)]
    elif action == ACTION_RIGHT:
        return [i, min(j + stepth,34)]
    else:
        assert False

agent = REINFORCE(30, 1, 4)

root_dir = 'ckpt_' + 'pg'
if not os.path.exists(root_dir):    
    os.mkdir(root_dir)


for i_episode in range(num_episodes):
    site = [20,25]
    state = torch.Tensor([im[20][25]])
    entropies = []
    log_probs = []
    rewards = []
    for t in range(num_steps):
        
        action, log_prob, entropy = agent.select_action(state)
        new_site = step(site, action,stepth)
        x0 = site[0]
        y0 = site[1]
        
        x1 = new_site[0]
        y1 = new_site[1]
        diff1 = im[x1][y1] - im[x0][y0]
#        print(diff1)
        
        
        if diff1 > 0:
            reward = 1 
        else:
            reward = -1 
        
        next_state = im[x1][y1]
        
        site = new_site
        state = next_state
        

        entropies.append(entropy)
        log_probs.append(log_prob)
        rewards.append(reward)
        state = torch.Tensor([next_state])
        if next_state == 255:
            break
#        if len(rewards) >= 2:
#            diff1 = rewards[-1]
#            diff2 = rewards[-2]
#            diff3 = diff1 - diff2
#            stepth += int(diff3)
            
#        if state == 255:
#            reward = 1
            
        
    #print(step)
    agent.update_parameters(rewards, log_probs, entropies, gamma)
    print(t,next_state)


    if i_episode % 50 == 0:
        torch.save(agent.model.state_dict(), os.path.join(root_dir, 'reinforce-'+str(i_episode)+'.pkl'))

    print("Episode: {}, reward: {}".format(i_episode, np.sum(rewards)))
