from torch import nn


class MLP(nn.Module):
    def __init__(self, input_units=3, hidden_units=4, bias=True):
        super(MLP, self).__init__()
        self.bias = bias

        self.fc1 = nn.Linear(input_units, hidden_units, bias=self.bias)
        self.fc2 = nn.Linear(hidden_units, 1, bias=self.bias)

        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()

        self.init(init_method="xavier")

    def init(self, init_method="uniform"):
        if init_method == "xavier":
            nn.init.xavier_uniform_(self.fc1.weight.data,
                                    gain=nn.init.calculate_gain('tanh'))
            nn.init.xavier_uniform_(self.fc2.weight.data,
                                    gain=1.0)
        else:
            nn.init.uniform_(self.fc1.weight.data, a=-.4, b=.4)
            nn.init.uniform_(self.fc2.weight.data, a=-.4, b=.4)
        # if self.bias:
        #     nn.init.xavier_uniform_(self.fc1.bias.data)
        #     nn.init.xavier_uniform_(self.fc2.bias.data)

    def forward(self, x):
        """ forward """
        out = self.tanh(self.fc1(x))
        out = self.fc2(out)
        return out
