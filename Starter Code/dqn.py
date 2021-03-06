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
    next_state = Variable(torch.FloatTensor(np.float32(next_state)), requires_grad=True)
    action = Variable(torch.LongTensor(action))
    reward = Variable(torch.FloatTensor(reward))
    done = Variable(torch.FloatTensor(done))
    # implement the loss function here

    QValue = model.forward(state).gather(1, action.unsqueeze(1)).squeeze(1)
    TargetQValues = target_model.forward(next_state).max(1)[0]

    ExpectedQValues = reward + gamma * TargetQValues * (1 - done)

    loss = (QValue - Variable(ExpectedQValues.data, requires_grad = True)).pow(2).mean()



    # ----------------------------------------
    
    # QValues = model.forward(state)          # gets current model q values
    # TargetQValues = target_model.forward(next_state) # gets current target model q values

    # NextStateQValues = TargetQValues.max(1)[0]      #gets highest q value in targe values
    # ExpectedQValues = reward + gamma * NextStateQValues * (1 - done) # calculates expected value with highest q value in next state

    # QValuesAtAction = QValues.gather(1, action.unsqueeze(1)).squeeze(1) # gets the q value at the action taken
    # loss = (QValuesAtAction - Variable(ExpectedQValues.data, requires_grad = True)).pow(2).mean() # calculates loss and turns loss into a tensor so be used in run_dqn_pong.py



    # ----------------------------------------

    # QValues = target_model(state).data      #all current state Q values of actions choosen
    # print("QValues = ", QValues)

    # TargetQValues = target_model(next_state)   #subsequent states Q values of actions choosen
    # print("TargetQValues = ", TargetQValues)

    # q_value = QValues.gather(1, action.unsqueeze(1))
    # print("q_value = ", q_value)

    # MaxQValue = TargetQValues.max(1)              # getting max Q value of next state of actions choosen
    # print("MaxQValue = ", MaxQValue)

    # ExpectedQValues = reward + gamma * MaxQValue
    # print("ExpectedQValue = ", ExpectedQValues)
    # loss = (q_value - Variable(ExpectedQValues.data, requires_grad = True)).pow(2).mean()

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

        # RandomSample = random.sample(self.buffer, batch_size)

        # state = []
        # action = []
        # reward = []
        # next_state = []
        # done = []
        
        # for frame in RandomSample:
        #     state.append(frame[0])
        #     action.append(frame[1])
        #     reward.append(frame[2])
        #     next_state.append(frame[3])
        #     done.append(frame[4])


        state, action, reward, next_state, done = zip(*random.sample(self.buffer, batch_size)) #takes random sample and organizes it into a single object that can be used to assign each corresponding variable
        state = np.concatenate(state) # need to make into a np array since td_loss uses it as such
        next_state = np.concatenate(next_state)



        return state, action, reward, next_state, done

    def __len__(self):
        return len(self.buffer)
