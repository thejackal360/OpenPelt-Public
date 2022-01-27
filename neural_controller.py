import numpy as np

import torch
from torch import nn
from torch.optim import Adam

from PySpice.Unit import u_V


class MLP(nn.Module):
    def __init__(self, hidden_units=4, bias=True):
        super(MLP, self).__init__()
        self.bias = bias

        self.fc1 = nn.Linear(2, hidden_units, bias=self.bias)
        self.fc2 = nn.Linear(hidden_units, 1, bias=self.bias)

        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

        self.init(init_method="xavier")

    def init(self, init_method="uniform"):
        if init_method == "xavier":
            nn.init.xavier_uniform_(self.fc1.weight.data,
                                    gain=nn.init.calculate_gain('relu'))
            nn.init.xavier_uniform_(self.fc2.weight.data,
                                    gain=nn.init.calculate_gain('sigmoid'))
        else:
            nn.init.uniform_(self.fc1.weight.data, a=-.01, b=.01)
            nn.init.uniform_(self.fc2.weight.data, a=-.01, b=.01)
        # if self.bias:
        #     nn.init.xavier_uniform_(self.fc1.bias.data)
        #     nn.init.xavier_uniform_(self.fc2.bias.data)

    def forward(self, x):
        """ forward """
        out = self.relu(self.fc1(x))
        out = self.sigmoid(self.fc2(out))
        return out


class neural_controller:
    def __init__(self, T_ref=-20.0, hidden_units=4, bias=True, lrate=0.01):
        self.t_ref = torch.FloatTensor([T_ref])
        self.net = MLP(hidden_units=hidden_units, bias=bias)
        self.optimizer = Adam(self.net.parameters(), lr=lrate)
        self.criterion = nn.MSELoss()

    def learn(self, t, T_hot, T_cold):
        x = torch.from_numpy(np.hstack([T_hot[-1], T_cold[-1]]).astype('f'))
        self.z = torch.tensor([T_cold[-1]], requires_grad=True)
        self.optimizer.zero_grad()
        yc = self.net(x)
        loss = self.criterion(self.z, self.t_ref)
        loss.backward()
        self.optimizer.step()
        yc = yc.detach().cpu().numpy()[-1] @ u_V
        print(yc, self.z, T_hot[-1], T_cold[-1], self.t_ref)
        return yc

    def controller(self, t, T_hot, T_cold, *args):
        x = torch.from_numpy(np.vstack([T_hot, T_cold]))
        yc = self.net(x)
        return yc
