import torch
import numpy as np
from torch.distributions import Categorical
from torch.autograd import Variable
from model import Policy


class VPG(object):
    def __init__(self, env, gamma, learning_rate):
        self.observation_space = env.observation_space.shape[0]
        self.action_space = env.action_space.n
        self.gamma = gamma
        self.policy = Policy(observation_space=self.observation_space, action_space=self.action_space)
        self.policy_history = Variable(torch.Tensor())
        self.reward_episode = []
        self.reward_history = []
        self.loss_history = []
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=learning_rate)

    def select_action(self, state):
        # Select an action (0 or 1) by running policy model and choosing based on the probabilities in state
        state = torch.from_numpy(state).type(torch.FloatTensor)
        state = self.policy(Variable(state))
        c = Categorical(state)
        action = c.sample()

        # Add log probability of our chosen action to our history
        self.policy_history = torch.cat([self.policy_history, c.log_prob(action).view(-1)])
        return int(action)

    def update_policy(self):
        R = 0
        rewards = []

        # Discount future rewards back to the present using gamma
        for r in self.reward_episode[::-1]:
            R = r + self.gamma * R
            rewards.insert(0, R)

        # Scale rewards
        rewards = torch.FloatTensor(rewards)
        rewards = (rewards - rewards.mean()) / (rewards.std() + np.finfo(np.float32).eps)

        # Calculate loss
        loss = (torch.sum(torch.mul(self.policy_history, Variable(rewards)).mul(-1), -1))

        # Update network weights
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Save and intialize episode history counters
        self.loss_history.append(loss.data)
        self.reward_history.append(np.sum(self.reward_episode))
        self.policy_history = Variable(torch.Tensor())
        self.reward_episode = []
