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
        self.input_units = input_units
        self.hidden_units = hidden_units

        self.fc1 = nn.Linear(input_units, hidden_units, bias=self.bias)
        self.fc2 = nn.Linear(hidden_units, hidden_units, bias=self.bias)
        self.fc3 = nn.Linear(hidden_units, hidden_units, bias=self.bias)
        self.fc4 = nn.Linear(hidden_units, output_units, bias=self.bias)

        self.layer_norm = nn.LayerNorm(hidden_units)

        self.relu = nn.ReLU()
        self.elu = nn.ELU()
        self.tanh = nn.Tanh()

        self.init(init_method="xavier")

    def init(self, init_method="uniform"):
        if init_method == "xavier":
            nn.init.xavier_uniform_(self.fc1.weight.data,
                                    gain=nn.init.calculate_gain('tanh'))
            nn.init.xavier_uniform_(self.fc2.weight.data)
            nn.init.xavier_uniform_(self.fc3.weight.data)
            nn.init.xavier_uniform_(self.fc4.weight.data)
        else:
            nn.init.trunc_normal_(self.fc1.weight.data, mean=0.0, std=1.0)
            self.fc1.weight.data /= self.input_units
            nn.init.trunc_normal_(self.fc2.weight.data, mean=0.0, std=1.0)
            self.fc2.weight.data /= self.hdden_units
            nn.init.trunc_normal_(self.fc3.weight.data, mean=0.0, std=1.0)
            self.fc3.weight.data /= self.hidden_units
            nn.init.trunc_normal_(self.fc4.weight.data, mean=0.0, std=1.0)
            self.fc4.weight.data /= 0.0001

        if self.bias is not None:
            self.fc1.bias.data.fill_(0)
            self.fc2.bias.data.fill_(0)
            self.fc3.bias.data.fill_(0)
            self.fc4.bias.data.fill_(0)

    def forward(self, x):
        """ forward """
        out = self.tanh(self.layer_norm(self.fc1(x)))
        out = self.elu(self.fc2(out))
        out = self.elu(self.fc3(out))
        out = self.fc4(out)
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
