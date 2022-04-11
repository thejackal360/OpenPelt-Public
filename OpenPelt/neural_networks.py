from torch import nn


class MLP(nn.Module):
    """
    This class implements a simple four layer MLP using Pytorch. This MLP is
    only for demonstration purposes.
    """
    def __init__(self,
                 input_units=3,
                 hidden_units=32,
                 output_units=12,
                 bias=True):
        """
        Initializes the neural network layers, activation functions, and
        parameters of the network such as weights and biases.

        @param input_units: number of units (or neurons) in the input layer

        @param hidden_units: number of neurons in the hidden layers

        @param output_units: number of neurons in the output layer (integer)

        @param bias: if it's true then bias terms will be added to each layer
        """

        super(MLP, self).__init__()
        self.bias = bias
        self.input_units = input_units
        self.hidden_units = hidden_units

        # Define four linear layers
        self.fc1 = nn.Linear(input_units, hidden_units, bias=self.bias)
        self.fc2 = nn.Linear(hidden_units, hidden_units, bias=self.bias)
        self.fc3 = nn.Linear(hidden_units, hidden_units, bias=self.bias)
        self.fc4 = nn.Linear(hidden_units, output_units, bias=self.bias)

        # Define a layer normalization
        self.layer_norm = nn.LayerNorm(hidden_units)

        # Define the non-linear activations
        self.elu = nn.ELU()
        self.tanh = nn.Tanh()

        # Initialize the weights and biases of all layers
        self.init(init_method="xavier")

    def init(self, init_method="uniform"):
        """
        Initializes the weights and biases of the neural network.

        @param init_method: the initialization method, it can be xavier (for
        a uniform Xavier method), trunc_normal for truncated normal
        distribution, or in any other case a uniform distribution.
        """
        if init_method == "xavier":
            nn.init.xavier_uniform_(self.fc1.weight.data)
            nn.init.xavier_uniform_(self.fc2.weight.data)
            nn.init.xavier_uniform_(self.fc3.weight.data)
            nn.init.xavier_uniform_(self.fc4.weight.data)
        elif "trunc_normal":
            nn.init.trunc_normal_(self.fc1.weight.data, mean=0.0, std=1.0)
            self.fc1.weight.data /= self.input_units
            nn.init.trunc_normal_(self.fc2.weight.data, mean=0.0, std=1.0)
            self.fc2.weight.data /= self.hdden_units
            nn.init.trunc_normal_(self.fc3.weight.data, mean=0.0, std=1.0)
            self.fc3.weight.data /= self.hidden_units
            nn.init.trunc_normal_(self.fc4.weight.data, mean=0.0, std=1.0)
            self.fc4.weight.data /= 0.0001
        else:
            nn.init.uniform_(self.fc1.weight.data, -0.01, 0.01)
            nn.init.uniform_(self.fc2.weight.data, -0.01, 0.01)
            nn.init.uniform_(self.fc3.weight.data, -0.01, 0.01)
            nn.init.uniform_(self.fc4.weight.data, -0.01, 0.01)

        if self.bias is not None:
            self.fc1.bias.data.fill_(0)
            self.fc2.bias.data.fill_(0)
            self.fc3.bias.data.fill_(0)
            self.fc4.bias.data.fill_(0)

    def forward(self, x):
        """ Pytorch forward method. This method runs the forward pass and
        retains all the information required for the backpropagation
        (gradients).

        @param x: input torch tensor of shape (N, L, Hin), where N is the
        batch size, L is any extra dimension such as a sequence length (L can
        be skipped), and Hin is the input dimension. In the case of a TEC
        neural controller N = 1, L is not used, and Hin = 2 since the input to
        the neural network is the actual temperature (sensor readout) and the
        reference temperature.

        @return A torch tensor of shape (N, Hout), where N is the batch size
        and Hout the output dimension. In the TEC neural controller case, N=1
        and Hout=1 reflecting the voltage value passed to the OpenPelt model.

        """
        out = self.tanh(self.layer_norm(self.fc1(x)))
        out = self.elu(self.fc2(out))
        out = self.elu(self.fc3(out))
        out = self.fc4(out)
        return out
