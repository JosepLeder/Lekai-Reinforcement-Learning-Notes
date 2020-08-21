import torch
import torch.nn as nn
from utils import init

class DiagGaussianDistribution(nn.Module):
    def __init__(self, num_inputs, num_outputs):
        super(DiagGaussianDistribution, self).__init__()

        init_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.
                               constant_(x, 0))

        self.fc_mean = init_(nn.Linear(num_inputs, num_outputs))
        self._bias = nn.Parameter(torch.zeros(num_outputs).unsqueeze(1))

    def forward(self, x):
        action_mean = self.fc_mean(x)

        #  An ugly hack for my KFAC implementation.
        zeros = torch.zeros(action_mean.size())

        if zeros.dim() == 2:
            bias = self._bias.t().view(1, -1)
        else:
            bias = self._bias.t().view(1, -1, 1, 1)
        action_logstd = zeros + bias

        return FixedNormal(action_mean, action_logstd.exp())


# Normal
class FixedNormal(torch.distributions.Normal):
    def log_probs(self, actions):
        return super().log_prob(actions).sum(-1, keepdim=True)

    def entrop(self):
        return super.entropy().sum(-1)

    def mode(self):
        return self.mean