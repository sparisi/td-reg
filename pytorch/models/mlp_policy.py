import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.normal import Normal
from torch.distributions.independent import Independent
from torch.distributions.multivariate_normal import MultivariateNormal
from utils.math import *
from utils import *


class Policy(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_size=(100, 100), activation='tanh', log_std=0, a_bound=1):
        super().__init__()
        self.is_disc_action = False
        if activation == 'tanh':
            self.activation = F.tanh
        elif activation == 'relu':
            self.activation = F.relu
        elif activation == 'sigmoid':
            self.activation = F.sigmoid
        elif activation == "leakyrelu":
            self.activation = F.leaky_relu

        self.affine_layers = nn.ModuleList()
        last_dim = state_dim
        for nh in hidden_size:
            self.affine_layers.append(nn.Linear(last_dim, nh))
            last_dim = nh

        self.action_mean = nn.Linear(last_dim, action_dim)
        self.action_mean.weight.data.mul_(0.1)
        self.action_mean.bias.data.mul_(0.0)
        self.a_bound = a_bound

        self.action_log_std = nn.Parameter(torch.ones(1, action_dim) * log_std)
        self.entropy_const = action_dim * ( 0.5 + 0.5 * torch.log(2 * torch.FloatTensor(1,1).fill_(math.pi))  ).to(device)

    def forward(self, x):
        for affine in self.affine_layers:
            x = self.activation(affine(x))

        action_mean = F.tanh(self.action_mean(x)) * self.a_bound
        action_log_std = self.action_log_std.expand_as(action_mean)
        action_std = torch.exp(action_log_std)

        return action_mean, action_log_std, action_std

    def select_action(self, x):
        action_mean, _, action_std = self.forward(x)
        action = torch.normal(action_mean, action_std)
        return action.data.view(-1)

    def select_greedy_action(self, x):
        action_mean, _, _ = self.forward(x)
        return action_mean.data.view(-1)

    def get_log_prob(self, x, actions):
        action_mean, action_log_std, action_std = self.forward(x)
        return normal_log_density(actions, action_mean, action_log_std, action_std)

    # need checking whether action_log_std change or not.
    def get_entropy(self):
        entropy = self.entropy_const + self.action_log_std.sum()
        return entropy

    def get_log_prob_entropy(self, x, actions):
        action_mean, action_log_std, action_std = self.forward(x)   #[batch_size, a_dim]

        normal = Normal(loc=action_mean, scale=action_std)
        diagn = Independent(normal, 1)
        log_prob = diagn.log_prob(actions).unsqueeze(dim=1)
        entropy = diagn.entropy()[0]

        #prob = MultivariateNormal(loc=action_mean, scale_tril=torch.diag(action_std[0,:]**2))
        #log_prob = prob.log_prob(actions).unsqueeze(dim=1)
        #entropy = prob.entropy()[0]

        return log_prob, entropy

    def get_fim(self, x):
        mean, _, _ = self.forward(x)
        cov_inv = self.action_log_std.data.exp().pow(-2).squeeze(0).repeat(x.size(0))
        param_count = 0
        std_index = 0
        id = 0
        for name, param in self.named_parameters():
            if name == "action_log_std":
                std_id = id
                std_index = param_count
            param_count += param.data.view(-1).shape[0]
            id += 1
        return cov_inv, mean, {'std_id': std_id, 'std_index': std_index}

class Policy_backup(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_size=(100, 100), activation='tanh', log_std=0, a_bound=1):
        super().__init__()
        self.is_disc_action = False
        if activation == 'tanh':
            self.activation = F.tanh
        elif activation == 'relu':
            self.activation = F.relu
        elif activation == 'sigmoid':
            self.activation = F.sigmoid
        elif activation == "leakyrelu":
            self.activation = F.leaky_relu

        self.affine_layers = nn.ModuleList()
        last_dim = state_dim
        for nh in hidden_size:
            self.affine_layers.append(nn.Linear(last_dim, nh))
            last_dim = nh

        self.action_mean = nn.Linear(last_dim, action_dim)
        self.action_mean.weight.data.mul_(0.1)
        self.action_mean.bias.data.mul_(0.0)
        self.a_bound = a_bound

        self.action_log_std = nn.Parameter(torch.ones(1, action_dim) * log_std)
        self.entropy_const = action_dim * ( 0.5 + 0.5 * torch.log(2 * torch.FloatTensor(1,1).fill_(math.pi))  ).to(device)

    def forward(self, x):
        for affine in self.affine_layers:
            x = self.activation(affine(x))

        action_mean = F.tanh(self.action_mean(x)) * self.a_bound
        action_log_std = self.action_log_std.expand_as(action_mean)
        action_std = torch.exp(action_log_std)

        return action_mean, action_log_std, action_std

    def select_action(self, x):
        action_mean, _, action_std = self.forward(x)
        action = torch.normal(action_mean, action_std)
        return action.data.view(-1)

    def select_greedy_action(self, x):
        action_mean, _, _ = self.forward(x)
        return action_mean.data.view(-1)

    def get_log_prob(self, x, actions):
        action_mean, action_log_std, action_std = self.forward(x)
        return normal_log_density(actions, action_mean, action_log_std, action_std)

    def get_entropy(self):
        entropy = self.entropy_const + self.action_log_std.sum()
        return entropy

    def get_fim(self, x):
        mean, _, _ = self.forward(x)
        cov_inv = self.action_log_std.data.exp().pow(-2).squeeze(0).repeat(x.size(0))
        param_count = 0
        std_index = 0
        id = 0
        for name, param in self.named_parameters():
            if name == "action_log_std":
                std_id = id
                std_index = param_count
            param_count += param.data.view(-1).shape[0]
            id += 1
        return cov_inv, mean, {'std_id': std_id, 'std_index': std_index}

class DiscretePolicy(nn.Module):
    def __init__(self, state_dim, action_num, hidden_size=(100, 100), activation='tanh'):
        super().__init__()
        self.is_disc_action = True
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

        self.action_head = nn.Linear(last_dim, action_num)
        self.action_head.weight.data.mul_(0.1)
        self.action_head.bias.data.mul_(0.0)

    def forward(self, x):
        for affine in self.affine_layers:
            x = self.activation(affine(x))

        action_prob = F.softmax(self.action_head(x), dim=-1)
        return action_prob

    def select_action(self, x):
        action_prob = self.forward(x)
        action = action_prob.multinomial(num_samples=1)
        return action.data.view(-1)

    def select_greedy_action(self, x):
        action_prob = self.forward(x)
        action = action_prob.argmax()
        return action.data.view(-1)

    def get_log_prob(self, x, actions):
        action_prob = self.forward(x)
        return torch.log(action_prob.gather(1, actions.unsqueeze(1)))

    def get_fim(self, x):
        action_prob = self.forward(x)
        M = action_prob.pow(-1).view(-1).data
        return M, action_prob, {}

