import torch.nn as nn
import torch.nn.functional as F


class Value(nn.Module):
    def __init__(self, state_dim, hidden_size=(100, 100), activation='tanh'):
        super().__init__()
        if activation == 'tanh':
            self.activation = F.tanh
        elif activation == 'relu':
            self.activation = F.relu
        elif activation == 'sigmoid':
            self.activation = F.sigmoid

        self.affine_layers = nn.ModuleList()
        last_dim = state_dim
        for nh in hidden_size:
            self.affine_layers.append(nn.Linear(last_dim, nh))
            last_dim = nh

        self.value_head = nn.Linear(last_dim, 1)
        self.value_head.weight.data.mul_(0.1)
        self.value_head.bias.data.mul_(0.0)

    def forward(self, x):
        for affine in self.affine_layers:
            x = self.activation(affine(x))

        value = self.value_head(x)
        return value

class Q_Value(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_size=(100, 100), activation='tanh'):
        super().__init__()
        if activation == 'tanh':
            self.activation = F.tanh
        elif activation == 'relu':
            self.activation = F.relu
        elif activation == 'sigmoid':
            self.activation = F.sigmoid

        self.fcs1 = nn.Linear(state_dim, hidden_size[0])
        self.fcs2 = nn.Linear(hidden_size[0], hidden_size[1])

        self.fca1 = nn.Linear(action_dim, hidden_size[1])
        #self.fc2 = nn.Linear(hidden_size[1], hidden_size[1])
        self.fc3 = nn.Linear(hidden_size[1], 1)

        self.fc3.weight.data.uniform_(-0.003, 0.003)


    def forward(self, state, action):
        s = self.activation(self.fcs1(state))
        s = self.fcs2(s)

        a = self.fca1(action)
        #x = torch.cat((s,a),dim=1)

        x = self.activation(s + a)
        #x = self.activation(self.fc2(x))

        x = self.fc3(x)

        return x

