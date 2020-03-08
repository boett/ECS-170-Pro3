from collections import deque
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.autograd as autograd
import math, random
USE_CUDA = torch.cuda.is_available()
Variable = lambda *args, **kwargs: autograd.Variable(*args, **kwargs).cuda() if USE_CUDA else autograd.Variable(*args, **kwargs)

class QLearner(nn.Module):
    def __init__(self, env, num_frames, batch_size, gamma, replay_buffer):
        super(QLearner, self).__init__()

        self.batch_size = batch_size
        self.gamma = gamma
        self.num_frames = num_frames
        self.replay_buffer = replay_buffer
        self.env = env
        self.input_shape = self.env.observation_space.shape
        self.num_actions = self.env.action_space.n

        self.features = nn.Sequential(
            nn.Conv2d(self.input_shape[0], 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU()
        )
        
        self.fc = nn.Sequential(
            nn.Linear(self.feature_size(), 512),
            nn.ReLU(),
            nn.Linear(512, self.num_actions)
        )
        
    def forward(self, x):   #throw input into forward function
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x
    
    def feature_size(self):
            return self.features(autograd.Variable(torch.zeros(1, *self.input_shape))).view(1, -1).size(1)
    
    def act(self, state, epsilon):  # will use the neural network to select whihc action to do based on what state
        if random.random() > epsilon:
            state = Variable(torch.FloatTensor(np.float32(state)).unsqueeze(0), requires_grad=True)
            # TODO: Given state, you should write code to get the Q value and chosen action


            action = torch.argmax(self(state)).item() # gets max Q value of the given state.





        else:
            action = random.randrange(self.env.action_space.n)
        #print("self = ", self)
        #print("------------------------------")
        #print("state = ", state)
        return action

    def copy_from(self, target):
        self.load_state_dict(target.state_dict())

        
def compute_td_loss(model, target_model, batch_size, gamma, replay_buffer): # computes the lose
    state, action, reward, next_state, done = replay_buffer.sample(batch_size)

    state = Variable(torch.FloatTensor(np.float32(state)))
    next_state = Variable(torch.FloatTensor(np.float32(next_state)).squeeze(1), requires_grad=True)
    action = Variable(torch.LongTensor(action))
    reward = Variable(torch.FloatTensor(reward))
    done = Variable(torch.FloatTensor(done))
    # implement the loss function here

    QValueTargetModel = target_model.forward(state).data
    loss = math.pow(reward - QValueTargetModel[0][action].item())
    
    return loss


class ReplayBuffer(object):
    def __init__(self, capacity):   #pulls frames to use for training on, randomly pulls from replay buffer so we dont minimize to a local minimum
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        state = np.expand_dims(state, 0)
        next_state = np.expand_dims(next_state, 0)

        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):

        #print("Sample self = ", self.buffer)
        # TODO: Randomly sampling data with specific batch size from the buffer

        #buffer = (self.buffer)

        #print("IN SAMPLE ")
        #print("sample self = ", self)

        pullFrom = self.buffer

        print("self.buffer[0][0] (state) = ", pullFrom[0][0])
        print("self.buffer[0][1] (action) = ", pullFrom[0][1].data)
        print("self.buffer[0][2] (reward) = ", pullFrom[0][2].data)
        print("self.buffer[0][3] (next_state) = ", pullFrom[0][3])
        print("self.buffer[0][4] (done) = ", pullFrom[0][4])

        state = []
        next_state = []
        action = []
        reward = []
        done = []

        for i in pullFrom:
            action.append(pullFrom[i][1])   
            reward.append(pullFrom[i][2])   
            next_state.append(pullFrom[i][3])
            done.append(pullFrom[i][4])

        return state, action, reward, next_state, done

    def __len__(self):
        return len(self.buffer)
