import numpy as np

import torch
from torch import nn
from torch.optim import Adam, SGD

from PySpice.Unit import u_V


class MLP(nn.Module):
    def __init__(self, input_units=3, hidden_units=4, bias=True):
        super(MLP, self).__init__()
        self.bias = bias

        self.fc1 = nn.Linear(input_units, hidden_units, bias=self.bias)
        self.fc2 = nn.Linear(hidden_units, 1, bias=self.bias)

        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

        self.init(init_method="uniform")

    def init(self, init_method="uniform"):
        if init_method == "xavier":
            nn.init.xavier_uniform_(self.fc1.weight.data,
                                    gain=nn.init.calculate_gain('relu'))
            nn.init.xavier_uniform_(self.fc2.weight.data,
                                    gain=nn.init.calculate_gain('relu'))
        else:
            nn.init.uniform_(self.fc1.weight.data, a=.0, b=.4)
            nn.init.uniform_(self.fc2.weight.data, a=.0, b=.4)
        # if self.bias:
        #     nn.init.xavier_uniform_(self.fc1.bias.data)
        #     nn.init.xavier_uniform_(self.fc2.bias.data)

    def forward(self, x):
        """ forward """
        out = self.relu(self.fc1(x))
        out = self.fc2(out)
        return out


class neural_controller:
    def __init__(self,
                 T_ref=-20.0,
                 input_units=3,
                 hidden_units=5,
                 bias=True,
                 lrate=0.01):
        adam = True
        # self.t_ref = torch.FloatTensor([T_ref])
        self.net = MLP(hidden_units=hidden_units, bias=bias)
        if adam:
            self.optimizer = Adam(self.net.parameters(),
                                  lr=lrate,
                                  weight_decay=1e-4)
        else:
            self.optimizer = SGD(self.net.parameters(),
                                 lr=lrate,
                                 weight_decay=1e-4,
                                 momentum=0.9)
        self.criterion = nn.MSELoss()
        self.loss_ = []

    def learn(self, T_hot, T_cold, T_ref, V):
        x = torch.from_numpy(np.array([T_hot,
                                       T_cold,
                                       T_ref]).astype('f'))
        self.t_ref = torch.tensor([T_ref], requires_grad=True)
        self.z = torch.tensor([T_cold], requires_grad=True)
        self.v = torch.tensor([V], requires_grad=True)

        self.optimizer.zero_grad()
        yc = self.net(x)
        # print(self.z.shape, self.t_ref.shape)
        # loss = self.criterion(self.z, self.t_ref)
        loss = self.criterion(yc, self.v)
        loss.backward()
        self.optimizer.step()

        # print("Target: %f, Pred: %f" % (self.v.item(), yc.item()))
        yc = yc.detach().cpu().numpy()[-1] @ u_V
        self.loss_.append(loss.item())
        return yc

    def controller(self, T_hot, T_cold, T_ref, *args):
        x = torch.from_numpy(np.array([T_hot,
                                       T_cold,
                                       T_ref]).astype('f'))
        yc = self.net(x)
        return np.round(yc.detach().cpu().numpy()[-1], 2)
