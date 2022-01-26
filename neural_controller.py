import numpy as np

import torch
from torch import nn
from torch.optim import Adam


class MLP(nn.Module):
    def __init__(self, hidden_units=4, bias=True):
        super(MLP, self).__init__()

        self.fc1 = nn.Linear(2, hidden_units, bias=bias)
        self.fc2 = nn.Linear(hidden_units, 1, bias=bias)
        self.relu = nn.ReLU()

    def forward(self, x):
        """ forward """
        out = self.relu(self.fc1(x))
        out = self.fc2(out)
        return out


class neural_controller:
    def __init__(self, T_ref=-20.0, hidden_units=4, bias=True, lrate=0.01):
        self.t_ref = torch.FloatTensor([T_ref])
        self.net = MLP(hidden_units=hidden_units, bias=bias)
        self.optimizer = Adam(self.net.parameters(), lr=lrate)
        self.criterion = nn.MSELoss()

    def learn(self, t, T_hot, T_cold):
        x = torch.from_numpy(np.hstack([T_hot[-1], T_cold[-1]]).astype('f'))
        self.optimizer.zero_grad()
        yc = self.net(x)
        loss = self.criterion(yc, self.t_ref)
        loss.backward()
        self.optimizer.step()
        return yc

    def controller(self, t, T_hot, T_cold, *args):
        x = torch.from_numpy(np.vstack([T_hot, T_cold]))
        yc = self.net(x)
        return yc
