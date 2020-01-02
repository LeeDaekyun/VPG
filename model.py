import torch
import torch.nn as nn


class Policy(nn.Module):
    def __init__(self, observation_space, action_space):
        super(Policy, self).__init__()
        self.l1 = nn.Linear(observation_space, 128, bias=False)
        self.l2 = nn.Linear(128, action_space, bias=False)

    def forward(self, x):
        model = torch.nn.Sequential(
            self.l1,
            nn.Dropout(p=0.6),
            nn.ReLU(),
            self.l2,
            nn.Softmax(dim=-1)
        )
        return model(x)
