import random
from torch import nn
from collections import namedtuple, deque


class MLP(nn.Module):
    def __init__(self,
                 input_units=3,
                 hidden_units=32,
                 output_units=12,
                 bias=True):
        super(MLP, self).__init__()
        self.bias = bias

        self.fc1 = nn.Linear(input_units, hidden_units, bias=self.bias)
        self.fc2 = nn.Linear(hidden_units, hidden_units, bias=self.bias)
        self.fc3 = nn.Linear(hidden_units, output_units, bias=self.bias)

        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()

        self.init(init_method="uniform")

    def init(self, init_method="uniform"):
        if init_method == "xavier":
            nn.init.xavier_uniform_(self.fc1.weight.data,
                                    gain=nn.init.calculate_gain('relu'))
            nn.init.xavier_uniform_(self.fc2.weight.data,
                                    gain=nn.init.calculate_gain('relu'))
            nn.init.xavier_uniform_(self.fc3.weight.data)
        else:
            nn.init.uniform_(self.fc1.weight.data, a=-.01, b=.01)
            nn.init.uniform_(self.fc2.weight.data, a=-.01, b=.01)
            nn.init.uniform_(self.fc3.weight.data, a=-.01, b=.01)

    def forward(self, x):
        """ forward """
        out = self.relu(self.fc1(x))
        out = self.relu(self.fc2(out))
        out = self.fc3(out)
        return out


Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))


class ReplayMemory():

    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)
