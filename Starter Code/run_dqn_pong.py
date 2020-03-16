from Wrapper.layers import *
from Wrapper.wrappers import make_atari, wrap_deepmind, wrap_pytorch
import math, random
import gym
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import torch.autograd as autograd 
import torch.nn.functional as F
USE_CUDA = torch.cuda.is_available()
from dqn import QLearner, compute_td_loss, ReplayBuffer

env_id = "PongNoFrameskip-v4"       # established environment that will be played
env = make_atari(env_id)
env = wrap_deepmind(env)
env = wrap_pytorch(env)

num_frames = 1000000        # total frames that will be learning from
batch_size = 32             # the number of samples that are provided to the model for update services at a given time
gamma = 0.99                # the discount of future rewards
record_idx = 10000          #

replay_initial = 10000      # number frames that are held
replay_buffer = ReplayBuffer(100000)
model = QLearner(env, num_frames, batch_size, gamma, replay_buffer)
model.load_state_dict(torch.load("pretrained_model.pth", map_location='cpu'))   #loading in the pretrained model

target_model = QLearner(env, num_frames, batch_size, gamma, replay_buffer)      #load in model
target_model.copy_from(model)

optimizer = optim.Adam(model.parameters(), lr=0.0001)      #learning rate set and optimizing the model
if USE_CUDA:
    model = model.cuda()            # sends model to gpu
    target_model = target_model.cuda()
    print("Using cuda")

epsilon_start = 1.0
epsilon_final = 0.01
epsilon_decay = 30000   #used in ?
epsilon_by_frame = lambda frame_idx: epsilon_final + (epsilon_start - epsilon_final) * math.exp(-1. * frame_idx / epsilon_decay)

losses = []
all_rewards = []
episode_reward = 0

state = env.reset() # initial state 

for frame_idx in range(1, num_frames + 1):  # plays until player or model gets score 21
    #print("Frame: " + str(frame_idx))      #uncomment to look at frames

    epsilon = epsilon_by_frame(frame_idx)
    action = model.act(state, epsilon)      #will write this function
    
    next_state, reward, done, _ = env.step(action)  #get next state
    replay_buffer.push(state, action, reward, next_state, done) #push actions resutls to buffer
    
    state = next_state
    episode_reward += reward
    
    if done:        # reset game and 
        state = env.reset()
        all_rewards.append((frame_idx, episode_reward))
        episode_reward = 0

    if len(replay_buffer) > replay_initial: #if number of plays has reached the limit calculate loss and optimize update model
        loss = compute_td_loss(model, target_model, batch_size, gamma, replay_buffer)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append((frame_idx, loss.data.cpu().numpy()))

    if frame_idx % 10000 == 0 and len(replay_buffer) <= replay_initial: #two ifs are just for printing
        print('#Frame: %d, preparing replay buffer' % frame_idx)

    if frame_idx % 10000 == 0 and len(replay_buffer) > replay_initial:
        print('#Frame: %d, Loss: %f' % (frame_idx, np.mean(losses, 0)[1]))
        print('Last-10 average reward: %f' % np.mean(all_rewards[-10:], 0)[1])

    if frame_idx % 50000 == 0:
        target_model.copy_from(model)   #updates target model
        print("saved")
        torch.save(model.state_dict(), "modelNew.pth")



