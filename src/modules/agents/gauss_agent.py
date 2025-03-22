from torch.distributions.normal import Normal


import torch.nn as nn
import torch.nn.functional as F


class GaussianAgent(nn.Module):
    def __init__(self, input_shape, args):
        super(GaussianAgent, self).__init__()
        self.args = args

        self.fc1 = nn.Linear(input_shape, args.hidden_dim)
        if self.args.use_rnn:
            self.rnn = nn.GRUCell(args.hidden_dim, args.hidden_dim)
        else:
            self.rnn = nn.Linear(args.hidden_dim, args.hidden_dim)
        # Output layer now produces mean and log_std for a Gaussian distribution
        self.fc_mean = nn.Linear(args.hidden_dim, args.n_actions)
        self.fc_log_std = nn.Linear(args.hidden_dim, args.n_actions)

    def init_hidden(self):
        # make hidden states on same device as model
        return self.fc1.weight.new(1, self.args.hidden_dim).zero_()

    def forward(self, inputs, hidden_state):
        x = F.relu(self.fc1(inputs))
        h_in = hidden_state.reshape(-1, self.args.hidden_dim)
        if self.args.use_rnn:
            h = self.rnn(x, h_in)
        else:
            h = F.relu(self.rnn(x))
        # Compute the mean and log_std for the Gaussian distribution
        mean = self.fc_mean(h)
        std = F.relu(self.fc_log_std(h))+1e-1
        return mean, std,h
